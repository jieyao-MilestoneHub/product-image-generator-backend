# 使用指南

1. ```git clone ```
2. ```cd product-image-generator-backend```
3. ```python -m venv venv```
4. ```./venv/Scripts/activate``` or ```source venv/Scripts/activate```
5. ```pip install requirements.txt```
6. 創建.env並把AWS key放入
```
AWS_ACCESS_KEY_ID=access_key_id
AWS_SECRET_ACCESS_KEY=secret_access_key
```
7. ```uvicorn main:app --upload```
8. 可以順利使用前端頁面