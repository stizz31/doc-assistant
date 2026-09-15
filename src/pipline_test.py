import torch
import torch.nn as nn
from vectorized import get_prepared_data
from model import TextClassifier

def main():
    print("Тестовый старт")

    X,y = get_prepared_data()

    in_features = X.shape[1] #это будет 1000
    num_classes = int(y.max())+1 #автоматически определит точное количество уникальных тем

    model = TextClassifier (in_features=in_features, num_classes=num_classes)
    print("\nСтруктура нейросети:")
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #Шаг А прыямой проход
    logits = model(X)
    #Шаг Б
    loss= loss_fn(logits, y)
    print("\n[ЗАМЕР] Значение ошибки До шага обчуения", loss.item())

    #Шаг В Обнулд\ение градиенов
    optimizer.zero_grad()

    #Шаг Г обратнвый проход
    optimazer.backward()

    #Шаг Д корректрировка весов
    optimazer.step()

    #Шаг контрольный замер
    new_logits = model(x)
    new_loss = loss_fn(new_logits, y)
    print("\n[ЗАМЕР] Значение ошибки До шага обчуения", new_loss.item())

    if new_loss.item() < loss.item():
        print("")

if __name__ =="__main___":
    main()
