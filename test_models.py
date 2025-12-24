import joblib
import pandas as pd
import numpy as np

def test_all_models():
    """Teste le chargement et la prédiction pour tous les modèles"""
    
    positions = ['attacker', 'midfielder', 'defender', 'goalkeeper']
    
    for position in positions:
        print(f"\n{'='*50}")
        print(f"Testing {position.upper()} model")
        print('='*50)
        
        try:
            # Charger le modèle
            model_data = joblib.load(f'models/best_model_{position}.pkl')
            
            print(f"✅ Modèle chargé : {type(model_data['model'])}")
            print(f"✅ Scaler : {type(model_data['scaler'])}")
            print(f"✅ Features ({len(model_data['features'])}):")
            for feat in model_data['features']:
                print(f"  - {feat}")
            
            # Créer des données de test
            test_data = {}
            for feat in model_data['features']:
                if 'age' in feat:
                    test_data[feat] = 25
                elif 'value' in feat:
                    test_data[feat] = 10.0
                elif 'per_match' in feat:
                    test_data[feat] = 1.0
                elif 'minutes' in feat:
                    test_data[feat] = 2000
                else:
                    test_data[feat] = 5.0
            
            # Créer DataFrame
            test_df = pd.DataFrame([test_data])
            test_df = test_df[model_data['features']]  # Bon ordre
            
            # Appliquer scaler
            if model_data['scaler']:
                X_test = model_data['scaler'].transform(test_df)
            else:
                X_test = test_df.values
            
            # Prédiction
            prediction = model_data['model'].predict(X_test)[0]
            print(f"✅ Prédiction test : {prediction:.2f} M€")
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_all_models()