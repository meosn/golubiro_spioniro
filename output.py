import csv
import os
print("\033c")
cc = ""
def read_csv():
    with open('compic.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        # next(reader)  # Пропускаем заголовок (если он есть)
        data = []
        for row in reader:
            if row:  # Игнорируем пустые строки
                data.append({
                    'name': row[0],
                    'comp': int(row[1])
                })
        return data

# Пример использования:
data = read_csv()
for item in data:
    print(f"Файл: {item['name']}, Оценка: {item['comp']}")
    cc = item['name']
    

folder_path = '/Users/mariasolovej/Documents/GitHub/golubiro_spioniro/images'  # Путь к папке
target_filename =  cc  # Имя файла, который ищем

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Проверяем наличие файла
if target_filename in files:
    print(f"Файл найден: {os.path.join(folder_path, target_filename)}")
else:
    print("Файл не найден.")