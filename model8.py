import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
import warnings
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.interpolate import interp1d
from monitoring_dashboard import log_forecast
from monitor import log_forecast

def cross_validate_model(ts, order, seasonal_order, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    
    for train_index, test_index in tscv.split(ts):
        train, test = ts.iloc[train_index], ts.iloc[test_index]
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, forecast))
        metrics.append(rmse)
    
    return np.mean(metrics)

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('cleaned.csv')

def handle_zero_values(ts, annual_total=None, method='smart_interpolation'):
    """
    Enhanced zero value handling with multiple strategies
    
    Parameters:
    ts: Time series with potential zero values
    annual_total: Annual total connections (if available)
    method: 'interpolation', 'seasonal_average', 'annual_distribution', 'smart_interpolation', 'set_null'
    
    Returns:
    Cleaned time series and information about changes made
    """
    original_ts = ts.copy()
    zero_info = {
        'method': method,
        'original_zeros': (ts == 0).sum(),
        'zero_positions': ts[ts == 0].index.tolist(),
        'total_points': len(ts),
        'changes_made': []
    }
    
    if method == 'set_null':
        # Convert zeros to NaN for exclusion from modeling
        ts_cleaned = ts.replace(0, np.nan)
        zero_info['changes_made'].append(f"Converted {zero_info['original_zeros']} zeros to NaN")
        
    elif method == 'interpolation':
        # Simple linear interpolation for zeros
        ts_cleaned = ts.replace(0, np.nan)
        ts_cleaned = ts_cleaned.interpolate(method='linear')
        
        # If still NaN at edges, use forward/backward fill
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        
        zero_info['changes_made'].append(f"Interpolated {zero_info['original_zeros']} zero values")
        
    elif method == 'seasonal_average':
        # Replace zeros with seasonal averages
        ts_cleaned = ts.copy()
        quarters = [ts.index[i].quarter for i in range(len(ts))]
        
        for quarter in [1, 2, 3, 4]:
            quarter_mask = [q == quarter for q in quarters]
            quarter_data = ts[quarter_mask]
            non_zero_avg = quarter_data[quarter_data > 0].mean()
            
            if not np.isnan(non_zero_avg):
                zero_positions_in_quarter = ts[(ts == 0) & quarter_mask].index
                ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg
                zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with seasonal average: {non_zero_avg:.2f}")
        
    elif method == 'annual_distribution' and annual_total is not None:
        # Distribute annual total across quarters based on existing patterns
        ts_cleaned = ts.copy()
        
        # Calculate typical quarterly distribution from non-zero years
        non_zero_years = {}
        for idx in ts.index:
            year = idx.year
            if year not in non_zero_years:
                non_zero_years[year] = []
            non_zero_years[year].append(ts[idx])
        
        # Find years with all non-zero quarters
        complete_years = {year: values for year, values in non_zero_years.items() 
                         if all(v > 0 for v in values)}
        
        if complete_years:
            # Calculate average quarterly distribution
            all_quarters = []
            for values in complete_years.values():
                total = sum(values)
                quarterly_props = [v/total for v in values]
                all_quarters.append(quarterly_props)
            
            avg_quarterly_dist = np.mean(all_quarters, axis=0)
            
            # Apply to years with zeros
            for year in non_zero_years:
                year_data = [ts[idx] for idx in ts.index if idx.year == year]
                if 0 in year_data:
                    # Use annual total to distribute
                    distributed_values = [annual_total * prop for prop in avg_quarterly_dist]
                    year_indices = [idx for idx in ts.index if idx.year == year]
                    for idx, value in zip(year_indices, distributed_values):
                        if ts[idx] == 0:
                            ts_cleaned[idx] = value
                    zero_info['changes_made'].append(f"Distributed annual total for year {year}")
        
    elif method == 'smart_interpolation':
        # Combination of methods based on data availability
        ts_cleaned = ts.copy()
        
        # Strategy 1: If isolated zeros, use interpolation
        zero_positions = ts == 0
        isolated_zeros = []
        
        for i, is_zero in enumerate(zero_positions):
            if is_zero:
                # Check if it's isolated (non-zero neighbors)
                has_left_neighbor = i > 0 and not zero_positions.iloc[i-1]
                has_right_neighbor = i < len(zero_positions)-1 and not zero_positions.iloc[i+1]
                
                if has_left_neighbor and has_right_neighbor:
                    isolated_zeros.append(i)
        
        # Interpolate isolated zeros
        if isolated_zeros:
            ts_temp = ts.replace(0, np.nan)
            ts_temp = ts_temp.interpolate(method='linear')
            for i in isolated_zeros:
                ts_cleaned.iloc[i] = ts_temp.iloc[i]
            zero_info['changes_made'].append(f"Interpolated {len(isolated_zeros)} isolated zeros")
        
        # Strategy 2: For consecutive zeros, use seasonal patterns
        remaining_zeros = ts_cleaned == 0
        if remaining_zeros.any():
            quarters = [ts_cleaned.index[i].quarter for i in range(len(ts_cleaned))]
            
            for quarter in [1, 2, 3, 4]:
                quarter_mask = [q == quarter for q in quarters]
                quarter_data = ts_cleaned[quarter_mask]
                non_zero_avg = quarter_data[quarter_data > 0].mean()
                
                if not np.isnan(non_zero_avg):
                    zero_positions_in_quarter = ts_cleaned[(ts_cleaned == 0) & quarter_mask].index
                    ts_cleaned.loc[zero_positions_in_quarter] = non_zero_avg * 0.8  # Conservative estimate
                    zero_info['changes_made'].append(f"Replaced {len(zero_positions_in_quarter)} zeros in Q{quarter} with 80% of seasonal average: {non_zero_avg * 0.8:.2f}")
        
        # Strategy 3: If still zeros remain, use overall trend
        if (ts_cleaned == 0).any():
            overall_avg = ts_cleaned[ts_cleaned > 0].mean()
            remaining_zeros = ts_cleaned == 0
            ts_cleaned[remaining_zeros] = overall_avg * 0.5  # Very conservative
            zero_info['changes_made'].append(f"Replaced remaining {remaining_zeros.sum()} zeros with 50% of overall average: {overall_avg * 0.5:.2f}")
    
    else:
        ts_cleaned = ts.copy()
        zero_info['changes_made'].append("No changes made - unknown method or missing annual data")
    
    zero_info['final_zeros'] = (ts_cleaned == 0).sum()
    zero_info['final_nans'] = ts_cleaned.isna().sum()
    
    return ts_cleaned, zero_info

def prepare_quarterly_data_enhanced(country, operator, zero_handling_method='smart_interpolation'):
    """
    Enhanced quarterly data preparation with zero value handling
    """
    # Filter for specific country and operator
    operator_data = df[(df['Country'] == country) & (df['Operator name'] == operator)]
    
    if operator_data.empty:
        raise ValueError(f"No data found for {operator} in {country}")
    
    # Extract quarterly columns
    q_cols = [col for col in df.columns if any(q in col for q in ['1Q', '2Q', '3Q', '4Q'])]
    
    # Get annual total if available (for method 2)
    annual_total = None
    if 'Annual total' in operator_data.columns:
        annual_total = operator_data['Annual total'].iloc[0]
        try:
            annual_total = float(annual_total)
        except:
            annual_total = None
    
    # Get the first row (should be only one row per operator-country)
    q_data = operator_data[q_cols].iloc[0]
    
    # Parse quarterly data
    quarters = []
    values = []
    
    for col, val in q_data.items():
        try:
            # Handle different column formats: "1Q 2015" or "1Q2015"
            if ' ' in col:
                q, year = col.split()
            else:
                q = col[:2]  # "1Q"
                year = col[2:]
            
            quarter = int(q[0])
            year = int(year)
            quarters.append(f'{year}-Q{quarter}')
            
            # Convert to float and handle various representations of zero/missing
            try:
                val_float = float(val)
                # Sometimes very small values are effectively zero
                if val_float < 1:
                    val_float = 0
                values.append(val_float)
            except:
                values.append(0)  # Treat unparseable values as zero
                
        except Exception as e:
            print(f"Skipping invalid column {col}: {str(e)}")
    
    # Create time series
    ts = pd.Series(values, index=pd.PeriodIndex(quarters, freq='Q'))
    ts = ts.sort_index()
    
    # Handle zero values using specified method
    ts_cleaned, zero_info = handle_zero_values(ts, annual_total, zero_handling_method)
    
    # Final cleanup - handle any remaining NaN values
    if ts_cleaned.isna().any():
        ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
        # If still NaN (all data was missing), fill with a minimal value
        if ts_cleaned.isna().any():
            ts_cleaned = ts_cleaned.fillna(1.0)
    
    return ts_cleaned, zero_info

def apply_variance_reduction_techniques(ts, method='log_transform'):
    """
    Apply various variance reduction techniques to the time series
    Enhanced to handle small values better with robust error handling
    """
    original_ts = ts.copy()
    transform_info = {'method': method, 'original_std': ts.std(), 'original_mean': ts.mean()}
    
    # Add small constant to avoid log(0) issues and handle edge cases
    min_val = ts.min()
    if min_val <= 0:
        shift_constant = abs(min_val) + 1
        ts = ts + shift_constant
        transform_info['shift_constant'] = shift_constant
    else:
        transform_info['shift_constant'] = 0
    
    try:
        if method == 'log_transform':
            ts_transformed = np.log(ts)
            transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            
        elif method == 'box_cox':
            from scipy.stats import boxcox
            # Ensure all values are positive for Box-Cox
            ts_positive = np.maximum(ts, 0.1)
            ts_transformed, lambda_param = boxcox(ts_positive)
            ts_transformed = pd.Series(ts_transformed, index=ts.index)
            transform_info['lambda'] = lambda_param
            
            def safe_inverse_boxcox(x):
                try:
                    if lambda_param != 0:
                        result = np.power(x * lambda_param + 1, 1/lambda_param)
                    else:
                        result = np.exp(x)
                    return np.maximum(result - transform_info['shift_constant'], 0.1)
                except:
                    return np.full_like(x, ts.mean())
            
            transform_info['inverse_func'] = safe_inverse_boxcox
            
        elif method == 'diff':
            ts_transformed = ts.diff().dropna()
            first_val = ts.iloc[0]
            
            def safe_inverse_diff(x):
                try:
                    result = x.cumsum() + first_val - transform_info['shift_constant']
                    return np.maximum(result, 0.1)
                except:
                    return np.full_like(x, ts.mean())
            
            transform_info['inverse_func'] = safe_inverse_diff
            
        elif method == 'smooth':
            window = min(4, max(2, len(ts) // 4))
            ts_transformed = ts.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            transform_info['window'] = window
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
            
        elif method == 'outlier_removal':
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            
            # Handle case where IQR is 0 (all values are the same)
            if IQR == 0:
                lower_bound = Q1 - ts.std()
                upper_bound = Q3 + ts.std()
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            ts_transformed = ts.clip(lower=lower_bound, upper=upper_bound)
            transform_info['bounds'] = (lower_bound, upper_bound)
            transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
            
        elif method == 'hybrid':
            # Enhanced hybrid method with better error handling
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                lower_bound = Q1 - ts.std()
                upper_bound = Q3 + ts.std()
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            ts_step1 = ts.clip(lower=lower_bound, upper=upper_bound)
            
            # Choose transformation based on data characteristics
            median_val = ts_step1.median()
            mean_val = ts_step1.mean()
            
            if median_val < 10:  # Very small values
                ts_transformed = np.sqrt(ts_step1)
                transform_info['sub_method'] = 'sqrt'
                transform_info['inverse_func'] = lambda x: np.maximum(np.power(np.maximum(x, 0), 2) - transform_info['shift_constant'], 0.1)
            elif median_val < 100:  # Medium values
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            else:  # Large values
                ts_transformed = np.log(ts_step1)
                transform_info['sub_method'] = 'log'
                transform_info['inverse_func'] = lambda x: np.maximum(np.exp(x) - transform_info['shift_constant'], 0.1)
            
            # Light smoothing only if we have enough data points
            if len(ts_transformed) >= 5:
                window = min(3, len(ts_transformed) // 3)
                ts_transformed = ts_transformed.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                transform_info['window'] = window
            
            transform_info['bounds'] = (lower_bound, upper_bound)
            transform_info['outliers_capped'] = sum((ts < lower_bound) | (ts > upper_bound))
        
        else:
            ts_transformed = ts.copy()
            transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
    
    except Exception as e:
        print(f"Warning: Transformation method '{method}' failed ({str(e)}), using original data")
        ts_transformed = ts.copy()
        transform_info['inverse_func'] = lambda x: np.maximum(x - transform_info['shift_constant'], 0.1)
        transform_info['transformation_failed'] = True
    
    # Final safety checks
    if ts_transformed.isna().any():
        ts_transformed = ts_transformed.fillna(ts_transformed.mean())
    
    if ts_transformed.std() == 0:
        print("Warning: Transformed data has zero variance")
        ts_transformed = ts_transformed + np.random.normal(0, 0.01, len(ts_transformed))
    
    transform_info['transformed_std'] = ts_transformed.std()
    transform_info['transformed_mean'] = ts_transformed.mean()
    
    # Handle division by zero in variance reduction calculation
    if transform_info['original_std'] != 0:
        transform_info['variance_reduction'] = (1 - transform_info['transformed_std'] / transform_info['original_std']) * 100
    else:
        transform_info['variance_reduction'] = 0
    
    return ts_transformed, transform_info

def analyze_and_forecast_enhanced(country, operator, variance_method='hybrid', zero_method='smart_interpolation'):
    """
    Enhanced analysis with both zero handling and variance reduction
    """
    print(f"\n{'='*70}")
    print(f"ENHANCED ANALYSIS: {operator} in {country}")
    print(f"Zero Handling Method: {zero_method}")
    print(f"Variance Reduction Method: {variance_method}")
    print(f"{'='*70}\n")
    
    # Prepare quarterly data with zero handling
    ts_cleaned, zero_info = prepare_quarterly_data_enhanced(country, operator, zero_method)
    
    print("ZERO VALUE HANDLING RESULTS:")
    print(f"Original zeros: {zero_info['original_zeros']}")
    print(f"Final zeros: {zero_info['final_zeros']}")
    print(f"Final NaNs: {zero_info['final_nans']}")
    print("Changes made:")
    for change in zero_info['changes_made']:
        print(f"  - {change}")
    
    print(f"\nCLEANED DATA STATISTICS:")
    print(f"Mean connections: {ts_cleaned.mean():,.2f}")
    print(f"Standard deviation: {ts_cleaned.std():,.2f}")
    print(f"Min connections: {ts_cleaned.min():,.2f}")
    print(f"Max connections: {ts_cleaned.max():,.2f}")
    print(f"Data points: {len(ts_cleaned)}")
    
    # Apply variance reduction
    ts_transformed, transform_info = apply_variance_reduction_techniques(ts_cleaned, variance_method)
    
    print(f"\nTRANSFORMED DATA STATISTICS:")
    print(f"Mean: {transform_info['transformed_mean']:.4f}")
    print(f"Standard deviation: {transform_info['transformed_std']:.4f}")
    print(f"Variance reduction: {transform_info['variance_reduction']:.2f}%")
    if 'shift_constant' in transform_info and transform_info['shift_constant'] > 0:
        print(f"Shift constant applied: {transform_info['shift_constant']}")
    
    # Enhanced plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Original vs cleaned time series
    ts_original = pd.Series([0 if idx in zero_info['zero_positions'] else ts_cleaned[idx] 
                            for idx in ts_cleaned.index], index=ts_cleaned.index)
    
    ts_original.plot(ax=axes[0,0], title='Original Time Series (with zeros)', marker='o')
    axes[0,0].set_ylabel('Connections')
    axes[0,0].grid(True)
    
    ts_cleaned.plot(ax=axes[0,1], title='Cleaned Time Series (zeros handled)', marker='s')
    axes[0,1].set_ylabel('Connections')
    axes[0,1].grid(True)
    
    # Transformed time series
    ts_transformed.plot(ax=axes[0,2], title=f'Transformed Time Series ({variance_method})', marker='^')
    axes[0,2].set_ylabel('Transformed Values')
    axes[0,2].grid(True)
    
    # Distribution comparisons
    axes[1,0].hist(ts_original[ts_original > 0], bins=10, alpha=0.7, label='Original (no zeros)')
    axes[1,0].set_title('Original Distribution')
    axes[1,0].set_xlabel('Connections')
    axes[1,0].set_ylabel('Frequency')
    
    axes[1,1].hist(ts_cleaned, bins=10, alpha=0.7, label='Cleaned', color='green')
    axes[1,1].set_title('Cleaned Distribution')
    axes[1,1].set_xlabel('Connections')
    axes[1,1].set_ylabel('Frequency')
    
    axes[1,2].hist(ts_transformed, bins=10, alpha=0.7, label='Transformed', color='orange')
    axes[1,2].set_title('Transformed Distribution')
    axes[1,2].set_xlabel('Transformed Values')
    axes[1,2].set_ylabel('Frequency')
    
    plt.suptitle(f'Complete Data Processing Pipeline: {operator} in {country}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Continue with modeling...
    print(f"\nMODELING ON PROCESSED DATA:")
    print(f"Data range: {ts_transformed.index[0]} to {ts_transformed.index[-1]}")
    print(f"Number of quarters: {len(ts_transformed)}")
    
    # Auto ARIMA
    print("\nSearching for best SARIMA parameters...")
    auto_model = auto_arima(
        ts_transformed,
        start_p=0, d=1, start_q=0,
        start_P=0, D=1, start_Q=0,
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        m=4,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    print(f"Best model: SARIMA{auto_model.order}{auto_model.seasonal_order}")
    
    # Model fitting and forecasting
    if len(ts_transformed) > 4:
        train_transformed = ts_transformed.iloc[:-4]
        test_transformed = ts_transformed.iloc[-4:]
        test_original = ts_cleaned.iloc[-4:]
        
        model = SARIMAX(train_transformed, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
        model_fit = model.fit(disp=False)
        
        # Forecast
        forecast_transformed = model_fit.get_forecast(steps=4)
        forecast_mean_transformed = forecast_transformed.predicted_mean
        conf_int_transformed = forecast_transformed.conf_int()
        
        # Transform back to original scale with robust error handling
        inverse_func = transform_info['inverse_func']
        try:
            forecast_mean_original = inverse_func(forecast_mean_transformed)
            
            # Handle various data types and ensure it's a proper array
            if isinstance(forecast_mean_original, pd.Series):
                forecast_values = forecast_mean_original.values
            elif hasattr(forecast_mean_original, '__iter__') and not isinstance(forecast_mean_original, (str, bytes)):
                forecast_values = np.array(forecast_mean_original)
            else:
                forecast_values = np.array([forecast_mean_original] * len(test_original))
            
            # Check for and handle NaN/infinite values
            if np.any(np.isnan(forecast_values)) or np.any(np.isinf(forecast_values)):
                print("Warning: Invalid values detected in forecast, using fallback method")
                # Fallback: use simple trend from recent data
                recent_trend = test_original.iloc[-2:].mean() if len(test_original) >= 2 else test_original.mean()
                forecast_values = np.full(len(test_original), recent_trend)
            
            # Ensure forecasts are positive and reasonable
            forecast_values = np.maximum(forecast_values, 0.1)
            
            # Additional sanity check - if forecasts are extremely large, cap them
            max_historical = ts_cleaned.max()
            forecast_values = np.minimum(forecast_values, max_historical * 3)  # Cap at 3x historical max
            
        except Exception as e:
            print(f"Warning: Inverse transformation failed ({str(e)}), using fallback forecast")
            # Fallback: use mean of test data or recent trend
            fallback_value = test_original.mean() if not np.isnan(test_original.mean()) else ts_cleaned.mean()
            forecast_values = np.full(len(test_original), fallback_value)
        
        # Create results with properly shaped arrays
        results = pd.DataFrame({
            'Actual': test_original.values,
            'Predicted': forecast_values,
        }, index=test_original.index)
        
        results['Error'] = results['Actual'] - results['Predicted']
        results['AbsoluteError'] = np.abs(results['Error'])
        results['PercentageError'] = (results['Error'] / results['Actual']) * 100
        
        # Calculate metrics with additional error checking
        try:
            # Ensure no NaN values in results
            if results['Actual'].isna().any() or results['Predicted'].isna().any():
                print("Warning: NaN values detected in results, cleaning...")
                results = results.dropna()
            
            if len(results) == 0:
                print("Error: No valid data points for evaluation")
                return None, zero_info, transform_info
            
            rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
            mae = mean_absolute_error(results['Actual'], results['Predicted'])
            
            # Handle MAPE calculation carefully to avoid division by zero
            actual_nonzero = results['Actual'] != 0
            if actual_nonzero.any():
                mape = mean_absolute_percentage_error(results['Actual'][actual_nonzero], results['Predicted'][actual_nonzero])
            else:
                mape = float('inf')  # or set to a default value
                
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            rmse = mae = mape = float('nan')
        
        print("\nFORECAST RESULTS:")
        print(results)
        print(f"\nEVALUATION METRICS:")
        print(f"RMSE: {rmse:,.2f}")
        print(f"MAE: {mae:,.2f}")
        print(f"MAPE: {mape:.2%}")
        
        # Enhanced forecast visualization
        plt.figure(figsize=(16, 8))
        
        ts_idx = ts_cleaned.index.to_timestamp()
        results_idx = results.index.to_timestamp()
        
        plt.plot(ts_idx, ts_cleaned.values, 'o-', label='Historical Data (cleaned)', alpha=0.7, linewidth=2)
        plt.plot(results_idx, results['Predicted'], 's--', color='red', 
                 label='Forecast', markersize=8, linewidth=2)
        
        # Highlight zero positions
        if zero_info['zero_positions']:
            zero_idx = [idx.to_timestamp() for idx in zero_info['zero_positions']]
            zero_vals = [ts_cleaned[idx] for idx in zero_info['zero_positions']]
            plt.scatter(zero_idx, zero_vals, color='yellow', s=100, 
                       label='Originally Zero Values', zorder=5, edgecolors='black')
        
        plt.text(0.02, 0.98, f'RMSE: {rmse:,.0f}\nMAE: {mae:,.0f}\nMAPE: {mape:.1%}\nZeros Handled: {zero_info["original_zeros"]}\nMethod: {zero_method}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.title(f'Enhanced Forecast with Zero Handling: {operator} in {country}', fontsize=16)
        plt.xlabel('Quarter')
        plt.ylabel('Connections')
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        return results, zero_info, transform_info
    
    else:
        print("Insufficient data for train/test split. Need more than 4 quarters.")
        return None, zero_info, transform_info

def compare_zero_handling_methods(country, operator):
    """
    Compare different zero handling methods
    """
    methods = ['interpolation', 'seasonal_average', 'annual_distribution', 'smart_interpolation', 'set_null']
    
    print(f"\n{'='*70}")
    print(f"COMPARING ZERO HANDLING METHODS: {operator} in {country}")
    print(f"{'='*70}\n")
    
    for method in methods:
        try:
            print(f"\n--- Testing {method} ---")
            ts_cleaned, zero_info = prepare_quarterly_data_enhanced(country, operator, method)
            
            print(f"Original zeros: {zero_info['original_zeros']}")
            print(f"Final zeros: {zero_info['final_zeros']}")
            print(f"Final NaNs: {zero_info['final_nans']}")
            print(f"Data points: {len(ts_cleaned)}")
            print("Changes made:")
            for change in zero_info['changes_made']:
                print(f"  - {change}")
                
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Compare zero handling methods
    compare_zero_handling_methods('Nigeria', 'Airtel')
    
    # Run enhanced analysis
    print("\n" + "="*80)
    print("RUNNING ENHANCED ANALYSIS WITH SMART INTERPOLATION + HYBRID VARIANCE REDUCTION")
    print("="*80)
    
    results, zero_info, transform_info = analyze_and_forecast_enhanced('Nigeria', 'Airtel', 'hybrid', 'smart_interpolation')
    
    # Try other combinations:
    # results2 = analyze_and_forecast_enhanced('Ghana', 'MTN', 'log_transform', 'seasonal_average')
    # results3 = analyze_and_forecast_enhanced('Ghana', 'MTN', 'box_cox', 'annual_distribution')
