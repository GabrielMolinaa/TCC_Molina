from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
import joblib 

def get_random_forest_model(X_train, y_train, save_path="rf_model.pkl"):
    rf_param_grid = {
        'n_estimators': (50, 500),          
        'max_depth': (5, 30),               
        'min_samples_split': (2, 20),       
        'min_samples_leaf': (1, 10)          
    }
    rf_bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=33, n_jobs=-1),
        search_spaces=rf_param_grid,
        n_iter=40,                         
        scoring='accuracy',                 
        cv=3,                               
        random_state=42,
        refit=True                         
    )
    print("Iniciando a busca pelos melhores hiperparâmetros para Random Forest...")
    rf_bayes_search.fit(X_train, y_train)
    
    best_params = rf_bayes_search.best_params_
    best_model = rf_bayes_search.best_estimator_
    best_score = rf_bayes_search.best_score_
    
    print(f"Melhores parâmetros encontrados para Random Forest: {best_params}")
    print(f"Melhor score de validação cruzada: {best_score:.4f}")
    
    joblib.dump(best_model, save_path)
    print(f"Modelo RF salvo em: {save_path}")

    return best_model, best_params, best_score
