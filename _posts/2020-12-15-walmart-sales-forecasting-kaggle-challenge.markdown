---
layout: post
title: Walmart Sales Forecasting Kaggle Challenge
date: 2020-12-15 01:00:00 +0300
description: I examine the calculations of ball park adjustments from the major stats websites. I believe ball park adjustments are often under-examined and methodologies need to be challeneged further. 
img: forecasting_thumbnail_9.png # Add image post (optional)
tags: [remote, remote work, remote jobs, work from home, future of work, the remote work era, women going remote, community] # add tag
---

```python
import os
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import joblib

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
```
##1. Setup
```python
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales.name = 'sales'
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar.name = 'calendar'
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
prices.name = 'prices'
```

```python
```

```python
```
