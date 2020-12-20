---
layout: post
title: Adjusted OPS Plus Workbook
date: 2019-08-23 01:00:00 +0300
description: Workbook compiles data from every play since 2008, create ball park factors, and create player stats for each season. Stats include traditional stats as well as newly created adjusted stats.
img: adjusted_ops_plus_thumbnail_12.png # Add image post (optional)
tags: [pitch deck, fundraising, entrepreneurship, startup, startups] # add tag
---
---
title: "Adjusted OPS+"
author: "Max Hoff"
date: "12/07/2020"
---

```r
#load packages

options("scipen"=100, "digits"=4)

library(readr)
library(stringr)
library(tidyr)
library(data.table)
library(dplyr)
library(tibble)
library(lubridate)
library(ggplot2)
```


```r
##combine all files into one dataframe

multi_union_all <- function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read_csv(file=x, col_names = c("category", "value_1", "value_2", "value_3", "value_4", "value_5", "value_6"))})
  Reduce(function(x,y) {union_all(x,y)}, datalist)}

combined_files_all_years <- multi_union_all("/Users/maxhoff/Downloads/Adjusted-OPS-Plus-main/data/play_by_play")
```

```r
##add in row number and game number

combined_files_all_years <- combined_files_all_years %>%
  add_column(row_num_orig_file = 1:nrow(combined_files_all_years), .before = "category") %>%
  mutate(game_number = ifelse(combined_files_all_years$category == "id",1,0)) %>%
  mutate(game_number = cumsum(game_number))
```

```r
##create separate dataframe for play log

plays_all_years <- combined_files_all_years %>%
  filter(category == "play")

plays_all_years <- plays_all_years %>%
  rename_all(recode, 
             value_1 = "inning",
             value_2 = "home/away",
             value_3 = "batter_id",
             value_4 = "ending_count",
             value_5 = "all_pitches",
             value_6 = "ab_result")
```

```r
##read play key in order to translate play descriptions into stats

play_key <- unique(read_csv("/Users/maxhoff/Downloads/Adjusted-OPS-Plus-main/data/play_key_v2.csv"))
```

```r
##merge with play log

play_key <- play_key %>%
  select(-hit_concat, -advancements_concat_sub, -advancements_concat, -X16)
  
plays_all_years <- merge(plays_all_years, play_key, by.x = "ab_result", by.y = "play", all.x = TRUE)
```

```r
##create game info log and merge with play log

game_info <- combined_files_all_years %>%
  filter(category == "info") %>%
  select(value_1, value_2, game_number) %>%
  spread(value_1, value_2)

plays_all_years <- merge(plays_all_years, game_info, by = "game_number",)

plays_all_years <- plays_all_years %>%
  mutate(season = str_sub(date, end = 4))
```

```r
##create TB field

plays_all_years$TB <- plays_all_years$`1B` + (plays_all_years$`2B` * 2) + (plays_all_years$`3B` * 3) + (plays_all_years$HR * 4)
```

```r
##load rosters in order to get handedness

folder_path = "/Users/maxhoff/Downloads/Adjusted-OPS-Plus-main/data/rosters_all_years"
file_list <- list.files(path = folder_path, full.names = TRUE) 

for (i in file_list){
  
  # if the merged dataset doesn't exist, create it
  if (!exists("rosters_all_years")){
    rosters_all_years <- read_csv(file = i, col_names = c("category", "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "season"))
    rosters_all_years[,"season"] <- str_sub(i, start = (nchar(i) - 7), end = (nchar(i) - 4))
  }
  
  # if the merged dataset does exist, append to it
  if (exists("rosters_all_years")){
    temp_dataset <- read_csv(file  = i, col_names = c("category", "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "season"))
    temp_dataset["season"] <- str_sub(i, start = (nchar(i) - 7), end = (nchar(i) - 4))
    rosters_all_years <- rbind(rosters_all_years, temp_dataset)
    rm(temp_dataset)
  }
}


rosters_all_years <- distinct(rosters_all_years)
```

```r
##reformat rosters df and merge with play log

rosters_all_years <- rosters_all_years %>%
  rename_all(recode, 
             category = "batter_id",
             value_1 = "last_name",
             value_2 = "first_name",
             value_3 = "bats",
             value_4 = "throws",
             value_5 = "team",
             value_6 = "position")

plays_all_years <- merge(plays_all_years, 
                         rosters_all_years[,c("batter_id", "last_name", "first_name", "bats", "team", "season")], 
                         by = c("batter_id", "season"), 
                         all.x = TRUE)
```

```r
##roll up play log by team by home and away games and by handedness in order to start creating ball park factors
##calculate slugging and obp
##difference in OPS between home and away fields by handedness will be basis of ball park factor calcs

##home stats df
home_stats <- plays_all_years %>%
  filter(`home/away` == "1") %>% 
  group_by(hometeam, bats) %>% 
  summarize(SLG_home = (sum(TB, na.rm = TRUE) / sum(AB, na.rm = TRUE)),
            OBP_home = ((sum(H, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE))
                   / (sum(AB, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE) + sum(SF, na.rm = TRUE))
                   )
            )
 
##away stats df
away_stats <- plays_all_years %>%
  filter(`home/away` == "0") %>% 
  group_by(visteam, bats) %>% 
  summarize(SLG_away = (sum(TB, na.rm = TRUE) / sum(AB, na.rm = TRUE)),
            OBP_away = ((sum(H, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE))
                   / (sum(AB, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE) + sum(SF, na.rm = TRUE))
                   )
            )

##merge home and away dfs

bpf_ops <- full_join(away_stats, 
                     home_stats, 
                     by = c("visteam" = "hometeam", "bats" = "bats")) %>% 
  rename_all(recode, 
             visteam = "team")

rm(home_stats, away_stats)
```

```r
##calc difference
##difference is ball park factor, i.e., how much to increase/decrease stats for each player

bpf_ops <- bpf_ops %>%
  mutate(SLG_adj = SLG_home - SLG_away,
         OBP_adj = OBP_home - OBP_away,
         OPS_adj = (SLG_home + OBP_home) - (SLG_away + OBP_home)) %>%
  select(-SLG_away, -OBP_away, -SLG_home, -OBP_home)
```

```r
##part 2
##create summary stats at the player level for the 2019 season

##home stats df
stats_by_player <- plays_all_years %>%
  group_by(batter_id, last_name, first_name, `home/away`, season, team, bats) %>% 
  summarize(ABs = sum(AB, na.rm = TRUE), 
            SLG = (sum(TB, na.rm = TRUE) / sum(AB, na.rm = TRUE)),
            OBP = ((sum(H, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE))
                   / (sum(AB, na.rm = TRUE) + sum(BB, na.rm = TRUE) + sum(HBP, na.rm = TRUE) + sum(SF, na.rm = TRUE))
                   )
            ) %>%
  filter(ABs > 0) %>%
  gather(stat, value, -(batter_id:bats)) %>%
  mutate(`home/away` = factor(ifelse(`home/away` == 0, "away", "home"))) %>%
  unite(temp, stat, `home/away`) %>%
  spread(temp, value) %>%
  mutate_all(~replace(., is.na(.), 0))
```

```r
##merge with bpf

stats_by_player <- left_join(stats_by_player, bpf_ops) %>%
  mutate(total_ABs = ABs_home + ABs_away,
         OPS_home = (OBP_home + SLG_home), 
         OPS_away = (OBP_away + SLG_away),
         adj_OBP_home = (OBP_home - OBP_adj),
         adj_SLG_home = (SLG_home - SLG_adj),
         adj_OPS_home = (OPS_home - OPS_adj),
         OBP = ((ABs_away / (ABs_away + ABs_home)) * OBP_away) + ((ABs_home / (ABs_away + ABs_home)) * OBP_home),
         SLG = ((ABs_away / (ABs_away + ABs_home)) * SLG_away) + ((ABs_home / (ABs_away + ABs_home)) * SLG_home),
         OPS = ((ABs_away / (ABs_away + ABs_home)) * OPS_away) + ((ABs_home / (ABs_away + ABs_home)) * OPS_home),
         adj_OBP = ((ABs_away / (ABs_away + ABs_home)) * OBP_away) + ((ABs_home / (ABs_away + ABs_home)) * adj_OBP_home),
         adj_SLG = ((ABs_away / (ABs_away + ABs_home)) * SLG_away) + ((ABs_home / (ABs_away + ABs_home)) * adj_SLG_home),
         adj_OPS = ((ABs_away / (ABs_away + ABs_home)) * OPS_away) + ((ABs_home / (ABs_away + ABs_home)) * adj_OPS_home),
         OBP_dif = adj_OBP - OBP,
         SLG_dif = adj_SLG - SLG,
         OPS_dif = adj_OPS - OPS
         ) %>%
  select(total_ABs, OBP, SLG, OPS, adj_OBP, adj_SLG, adj_OPS, OBP_dif, SLG_dif, OPS_dif) %>%
  dplyr::mutate(OPS_rank = dense_rank(desc(OPS)),
         adj_OPS_rank = dense_rank(desc(adj_OPS)),
         .before = "total_ABs")
```

```r
##EDA

stats_by_player %>% 
  filter(total_ABs > 500,
         season == '2019') %>%
  arrange(desc(OPS))
```
