from sklearn.svm import SVC
from skopt import BayesSearchCV
import joblib 

def get_svm_model(X_train, y_train, save_path="svm_model.pkl"):
    svm_param_grid = {
        'C': (0.1, 10.0, 'log-uniform'),
        'gamma': (0.01, 1.0, 'log-uniform'),
        'kernel': ['rbf']
    }
    svm_bayes_search = BayesSearchCV(
        estimator=SVC(random_state=33),
        search_spaces=svm_param_grid,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        random_state=33
    )
    print("Iniciando a busca pelos melhores hiperparâmetros para SVM...")
    svm_bayes_search.fit(X_train, y_train)
    

    best_params = svm_bayes_search.best_params_
    best_model = svm_bayes_search.best_estimator_
    
    print(f"Melhores parâmetros encontrados para SVM: {best_params}")
    
    
    joblib.dump(best_model, save_path)
    print(f"Modelo SVM salvo em: {save_path}")
    
    return best_model, best_params
