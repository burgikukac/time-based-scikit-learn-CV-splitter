# Time-based Cross-Validation with Scikit-Learn

While running a **FLAML autoML job**, I realized I needed a proper **time-based cross-validation** for my Polars dataframe. FLAML can use Scikit-Learn splitter objects, but the ones provided didn't meet my requirements. So, I quickly wrote my own.

## What is this?
A time-based cross-validation where the window size is fixed in terms of **time** rather than data points. You can use **pandas** and **polars** dataframes and **numpy** arrays as input. 

## Why use it?
To achieve a **better cross-validation estimation**.

## When to use it?
- If your data has a **temporal component**.
- If your **test/inference data** will contain **newer data** than your training data.

In this case, we want to emulate/estimate the test distribution/performance. For example, if we know the test data will cover the next **X minutes/days/months**, then we can use this approach in cross-validation.

**Examples**: housing price estimation, forecasting events based on logs, etc.

## How to use it?

### ⚠️ Limitation of Scikit-Learn’s `TimeSeriesSplit`:
- It's designed for **time series data** that features regular intervals, no duplicates, and no missing entries—quite restrictive.
- It also creates **fixed-size windows** in terms of row numbers (analogy: row-based window functions in SQL). This means data from the same time period could be present in both training and validation sets, which is not ideal (though still better than nothing).

### ✅ Integer-based time column:
- If your time column is already an **integer** (e.g., **UNIX days** or another form that maintains temporal order) and you can easily express the window and training sizes in those values, then you can implement a **custom splitter object** with a generator function. This is similar to **range-based window functions** in SQL using integers.
- However, this type of implementation isn't directly supported in Scikit-Learn. One reason might be that implementing this feature requires access to the time variable, which could break certain assumptions in the framework.

### ❓ Other scenarios:
- If the window size cannot be easily defined based on the time variable (e.g., "5 workdays" where "day" is the basis), you may need to **make a compromise**.
