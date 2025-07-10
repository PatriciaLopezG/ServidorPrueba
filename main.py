from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
import pandas as pd
from pydantic import BaseModel



# Carga del modelo entrenado (fuera de las funciones para optimizar)
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    # 1. Validar que vengan todas las características necesarias
    required = ["Pclass", "Sex", "Age", "Fare"]
    missing = [f for f in required if f not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan características: {', '.join(missing)}"
        )

    # 2. Convertir 'Sex' de texto a numérico (0=male, 1=female)
    sex = data["Sex"]
    if isinstance(sex, str):
        s = sex.lower()
        if s == "male":
            sex_num = 0
        elif s == "female":
            sex_num = 1
        else:
            raise HTTPException(
                status_code=400,
                detail="Valor de 'Sex' inválido (debe ser 'male' o 'female')"
            )
    elif isinstance(sex, (int, float)):
        sex_num = int(sex)
    else:
        raise HTTPException(
            status_code=400,
            detail="Tipo de 'Sex' inválido"
        )

    # 3. Construir el DataFrame con tipos adecuados
    try:
        row = {
            "Pclass": int(data["Pclass"]),
            "Sex": sex_num,
            "Age": float(data["Age"]),
            "Fare": float(data["Fare"])
        }
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al convertir datos: {e}"
        )

    df = pd.DataFrame([row])

    # 4. Hacer la predicción y devolver un JSON sencillo
    pred = model.predict(df)
    survived = int(pred[0])
    return {"survived": survived}



