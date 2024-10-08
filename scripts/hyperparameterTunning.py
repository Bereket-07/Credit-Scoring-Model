import logging ,os
from sklearn.model_selection import GridSearchCV


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


def hyperparameter_for_logisticModel(model,X_train,y_train):
    logger.info("hyper parameter Tunning for Logistic Regression model")
    try:
        logger.info("defining the logistic param grid")
        logistic_param_grid = {
            'C':[0.001,0.01,0.1,1,10],
            'solver':['liblinear','lbfgs']
        }
        logger.info("Applying the GridSearchCV for Logistic Regression model")
        logistic_grid_search = GridSearchCV(model,logistic_param_grid,scoring='accuracy',cv=5)
        logger.info("fiting the logistic grid search")
        logistic_grid_search.fit(X_train,y_train)

        logger.info("searching for the best logistic model estimator")
        best_logistic_model = logistic_grid_search.best_estimator_
        print("Best Logistic Regression Params:", logistic_grid_search.best_params_)
        return best_logistic_model
    except Exception as e:
        logger.error(f"error occured {e}")
def hyperparameter_for_randomForest(model,X_train,y_train):
    logger.info("Applying the GridSearch for random forest hyperparameter tunning")
    try:
        logger.info("difining the random forest parameter grid")
        rf_param_grid = {
            'n_estimators':[50,100,200],
            'max_depth':[None,10,20,30],
            'min_samples_split':[2,5,10],
        }
        logger.info("fiting the Random Forest model")
        rf_grid_search = GridSearchCV(model,rf_param_grid,scoring='accuracy', cv=5)
        logger.info("fiting the random Forest grid search")
        rf_grid_search.fit(X_train,y_train)
        logger.info("searching for the best random forest model estimator")
        best_rf_model = rf_grid_search.best_estimator_
        print("Best Random Forest Params:", rf_grid_search.best_params_)
        return best_rf_model
    except Exception as e:
        logger.error(f"error occured {e}")
        return None


