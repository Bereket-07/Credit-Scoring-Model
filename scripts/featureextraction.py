import logging , os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , MinMaxScaler, StandardScaler
from xverse.transformer import WOE
import datetime as dt
import matplotlib.pyplot  as plt

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format of the log messages
)
# Create a logger object
logger = logging.getLogger(__name__)

# define the path to the Logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','logs')

# create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)


def load_data(path):
    logger.info("loading the data")
    try:
        logger.info("the data is loding")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"error occured {e}")
        return None
def creating_aggregate_features(data):
    logger.info("creating aggregate features")
    try:
        aggregates = data.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount','sum'), # sum of all transaction amounts per customer
            Average_Transaction_Amount=('Amount','mean'), #Average transaction amount per customer
            Transaction_Count = ('TransactionId','count'), # count of transaction per customer
            Std_Transaction_Amount = ('Amount','std')  # standard deviation of tansaction amount per customer
        ).reset_index()

        # Merge the aggregated features back into the original dataframe 'df'
        data = data.merge(aggregates, on='CustomerId', how='left')
        return data
    except Exception as e:
        logger.error(f"error occured {e}")
def extract_features(data):
    logger.info("extracing some features")
    try:
        # Assuming the 'TransactionStartTime' is in string format, let's convert it to datetime
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # Extracting features from 'TransactionStartTime'
        data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour     # Hour of the transaction
        data['Transaction_Day'] = data['TransactionStartTime'].dt.day       # Day of the month
        data['Transaction_Month'] = data['TransactionStartTime'].dt.month   # Month of the year
        data['Transaction_Year'] = data['TransactionStartTime'].dt.year     # Year of the transaction
        return data
    except Exception as e:
        logger.error(f"error occured {e}")
def encoding(data, target_variable='FraudResult'):
    logger.info("encoding the categorical variables")
    try:
        # Apply Label Encoding for ordinal categorical variables first
        logger.info("label encoding for ordinal categorical variables")
        # (Your existing label encoding code)

        # Check data types of columns for WOE encoding
        logger.info("Checking data types of columns for WOE encoding")
        logger.info(data.dtypes)

        # Inspect unique values before conversion
        for col in ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']:
            logger.info(f"Unique values in {col}: {data[col].unique()}")

        # Convert categorical to numeric using category codes
        for col in ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']:
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes

        logger.info("Data types after conversion:")
        logger.info(data.dtypes)

        # Now apply WOE encoding
        logger.info("applying WOE encoding to certain categorical variables...")
        woe = WOE()
        try:
            woe.fit(data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']], data[target_variable])
            data_woe = woe.transform(data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']])
            data = pd.concat([data, data_woe], axis=1)
            iv_values = woe.iv_values_
            logger.info(f"Information Value (IV) for features: {iv_values}")
        except Exception as e:
            logger.error(f"Error during WOE transformation: {e}")

        # Now apply one-hot encoding for nominal variables
        logger.info("one-hot encoding for nominal variables")
        # (Your existing one-hot encoding code)

        return data

    except Exception as e:
        logger.error(f"error occurred {e}")
        return data  # Return original data on error

def Standardize_numeical_features(data):
    logger.info("normalize the numerical features")
    try:
        numerical_features = ['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount']
        # Normalize the Numerical Features (Range [0,1])
        # Initialize MinMaxScalar for normalization
        min_max_scalar =MinMaxScaler()

        # # Apply normalization to the numerical columns
        # data[numerical_features] = min_max_scalar.fit_transform(data[numerical_features])

        # logger.info(f"the result of the nirmalized numeical featuresis  \n {data[numerical_features].head()}")

        # Standardize the Numerical Featutres (Mean 0 , Standard Deviation 1)
            #   initialize StandardScalar for standardixation
        standard_scalar = StandardScaler()
        # Apply standardization to the numerical columns
        data[numerical_features] = standard_scalar.fit_transform(data[numerical_features])

        # check the result 
        logger.info(f"the result of the standardized numeical featuresis  \n {data[numerical_features].head()}")
        return data
    except Exception as e:
        logger.error(f"error occured {e}")
def constructinf_RFMS_scores(data):
    logger.info("constructing the RFMS scores")
    try:
        logger.info("Calculate Recency as days since last transaction")
        
        # Convert TransactionStartTime to datetime
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # Get the current date in UTC
        current_date = dt.datetime.now(dt.timezone.utc)

        # Calculate Recency as days since the last transaction
        data['Recency'] = (current_date - data['TransactionStartTime']).dt.days

        # Creating RFMS score; weight components based on their importance
        data['RFMS_score'] = (1 / (data['Recency'] + 1) * 0.4) + (data['Transaction_Count'] * 0.3) + (data['Total_Transaction_Amount'] * 0.3)
        
        logger.info("visualizing the RFMS space")
        visualuze_RFMS_space(data)
        
        logger.info("assigning the good and bad labels")
        assign_good_and_bad_lables(data)
        
        logger.info("calculating woe")
        # Ensure that the 'Label' and 'RFMS_score' columns exist in the DataFrame
        if 'Label' in data.columns and 'RFMS_score' in data.columns:
            woe_results = calculate_woe(data, 'Label', 'RFMS_score')
            logger.info("WoE results calculated successfully")
        else:
            logger.error("Columns 'Label' or 'RFMS_score' not found in DataFrame")
            return

        # Print WoE results
        print("WoE Results:")
        print(woe_results)
        
        return data
    
    except Exception as e:
        logger.error(f"error occurred {e}")

def visualuze_RFMS_space(data):
    try:
        # Scatter plot of RFMS scores
        plt.scatter(data['Transaction_Count'], data['Total_Transaction_Amount'], c=data['RFMS_score'], cmap='viridis')
        plt.colorbar(label='RFMS Score')
        plt.xlabel('Transaction Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('RFMS Visualization')
        plt.show()
    except Exception as e:
        logger.error(f"Error in visualizing RFMS space: {e}")

def assign_good_and_bad_lables(data):
    # Define threshold
    threshold = data['RFMS_score'].median()
    
    # Assign Good (1) and Bad (0) labels based on the threshold
    data['Label'] = (data['RFMS_score'] > threshold).astype(int)

def calculate_woe(df, target, feature):
    """
    Calculate the Weight of Evidence (WoE) for a given feature.
    """
    try:
        # Group by the feature and calculate the counts and events
        woe_df = df.groupby(feature)[target].agg(
            count=('Label', 'size'), 
            event=('Label', 'sum')
        ).reset_index()

        woe_df['non_event'] = woe_df['count'] - woe_df['event']
        
        # Check for zero division
        total_events = woe_df['event'].sum()
        total_non_events = woe_df['non_event'].sum()

        # Avoid division by zero
        if total_events == 0 or total_non_events == 0:
            raise ValueError("Total events or non-events cannot be zero.")

        woe_df['event_rate'] = woe_df['event'] / total_events
        woe_df['non_event_rate'] = woe_df['non_event'] / total_non_events
        
        # Calculate WoE
        woe_df['woe'] = np.log(woe_df['event_rate'] / woe_df['non_event_rate']).replace([-np.inf, np.inf], 0)
        
        return woe_df[[feature, 'count', 'event', 'non_event', 'woe']]
    
    except Exception as e:
        logger.error(f"Error in calculating WoE: {e}")

