# ebw_prediction_model
Educational ML project to choose a model

Датасет используемый для обучения и проверки модели имеет 72 строки.
Из-за минимального разброса данных SMOTE приводит к ументшению точности модели (скорее всего из-за эффекта overfitting)
В связи с этим модель для использования в обучения и использования в приложении будет выбираться из двух регрессионных моделей:
  - Decision Tree
  - Random Forest
Также будет изучена возможность улучшения результата при использовании стандартизации и нормализации
В данном проекте осознано не используется принцип DRY для наглядности учебног проекта.
