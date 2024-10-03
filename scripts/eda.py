import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging , os


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
    logger.info("loading the data set into the pandas data frame")
    try:
        logger.info("the data set loaded succesfully")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"error while loading the data : {e}")
        return None
def overview_the_data(data):
    logger.info("showing the overview of he data")
    try:
        rows , cols = data.shape
        logger.info(f"the data set have {cols} columns and {rows} rows")
        logger.info(f"the data set columns data types \n {data.dtypes}")
    except Exception as e:
        logger.error(f"error occures {e}")
def summary_statistics(data):
    logger.info("trying to perfomr the summary statistics")
    try:
        logger.info(f"the statistical summaery is \n {data.describe()}")
    except Exception as e:
        logger.error(f"error occuerd in summary statistics {e}")
# the ditribution analysis for the numeric cols
def distribution_of_numerical_features(data):
    logger.info("perfomring the process to show the distribution of numerical features")
    try:
        logger.info("selecting columns of the numeric columns only")
        numeric_cols = data.select_dtypes(include = ['float64', 'int64']).columns
        logger.info("visualing the numeric data")
        visualize_distribution(data,numeric_cols)
        logger.info("box plot for outliers for numeric columns")
        box_plot_for_outliers_detections(data,numeric_cols)
        pair_plots_for_Multivraiate_analysis(data,numeric_cols)
        check_for_skewness(data,numeric_cols)
    except Exception as e:
        logger.error(f"error occured {e}")
def visualize_distribution(data,numeric_cols):
    plt.figure(figsize=(15,10))
    for i , col in enumerate(numeric_cols,1):
        plt.subplot(3,3,i) # Adjust the layout based on the number of numerical columns
        plt.hist(data[col],bins=30,edgecolor='black')
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
def box_plot_for_outliers_detections(data,numeric_cols):
    for i , col in enumerate(numeric_cols,1):
        plt.subplot(3,3,i) # Adjust the layout based on the number of numrtical columns
        plt.boxplot(data[col])
        plt.title(col)
        plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
# to obsserver rhe relationship between multiple numerical features adnidentify any correlation patterns i perform pair plots
def pair_plots_for_Multivraiate_analysis(data,numeric_cols):
    # generating the pair plots for numerical columns
    sns.pairplot(data[numeric_cols])
    plt.show()
def check_for_skewness(data,numeric_cols):
    skewness = data[numeric_cols].skew()
    print("Skewnwss for Numerical Features:\n", skewness)

    # visualize skewness with a bar plot
    plt.figure(figsize=(10,5))
    skewness.plot(kind='bar')
    plt.title('Skewness of numerical Features')
    plt.xlabel('Features')
    plt.ylabel('Skewness')
    plt.axhline(0,color='red',linestyle='--')
    plt.show()
# Distribution analysis for the catagorical cols
def distribution_of_catagorical_features(data):
    logger.info("perfomring the process to show the distribution of catagorical features")
    try:
        catagorical_cols = data.select_dtypes(include = ['object', 'category']).columns
        visualize_distribution_catagoical(data,catagorical_cols)
        pie_charts_for_proportions(data,catagorical_cols)
        analyze_the_variability_of_catagorical_features(data,catagorical_cols)
    except Exception as e:
        logger.error(f"error occured {e}")
def visualize_distribution_catagoical(data,catagorical_cols):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(catagorical_cols, 1):
        plt.subplot(3, 3, i)  # Adjust based on the number of categorical columns
        data[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel('Categories')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
def pie_charts_for_proportions(data,catagorical_cols):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(catagorical_cols, 1):
        plt.subplot(3, 3, i)
        data[col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f'Proportion of {col}')
        plt.ylabel('')  # Hide y-label for better visibility
    plt.tight_layout()
    plt.show()
def analyze_the_variability_of_catagorical_features(data,catagorical_cols):
    # Number of unique categories in each categorical column
    unique_counts = data[catagorical_cols].nunique()
    print("Number of unique categories per categorical feature:\n", unique_counts)

    # Visualizing variability with a bar plot
    plt.figure(figsize=(10, 5))
    unique_counts.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Number of Unique Categories in Categorical Features')
    plt.xlabel('Categorical Features')
    plt.ylabel('Number of Unique Categories')
    plt.show()
def correlation_matrix_for_numerical_features(data):
    logger.info("performing the correlation matrix for the numeric features")
    try:
        numeric_cols = data.select_dtypes(include = ['float64', 'int64']).columns
        numeric_df = data[numeric_cols]
        correlation(numeric_df,numeric_cols)
        print(corr_visualization_heatmap(correlation(numeric_df,numeric_cols)))
        pairwise_scatter_plots_for_detailed_rl(numeric_df)
        higly_correlated_features(correlation(numeric_df,numeric_cols))
    except Exception as e:
        logger.error(f"error occured {e}")

def correlation(numeric_df,numeric_cols):
    correlation_matrix = numeric_df.corr(method='pearson')
    return correlation_matrix
def corr_visualization_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
def pairwise_scatter_plots_for_detailed_rl(numeric_df):
    sns.pairplot(numeric_df)
    plt.show()
def higly_correlated_features(correlation_matrix):
    # Get highly correlated pairs (absolute value > 0.7)
    high_corr = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
    print("Highly correlated features:\n", high_corr)

#  see for missing values
def check_for_missing_values(data):
    # Check for missing values in the dataset
    missing_values = data.isnull().sum()

    # Display columns with missing values
    missing_values = missing_values[missing_values > 0]
    print(missing_values)

def outlier_detection_for_numeric_cols(data):
    logger.info("detecting outliers for numerical columns")
    try:
        numeric_cols = data.select_dtypes(include = ['float64', 'int64']).columns
        numeric_df = data[numeric_cols]
        # Box plot for all numerical columns
        sns.boxplot(data=numeric_df)
        plt.xticks(rotation=90)  # Rotate labels if needed
        plt.title('Box Plot