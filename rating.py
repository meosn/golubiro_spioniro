import telebot
from telebot import types
import uuid
from PIL import Image
import os
from io import BytesIO
import csv
import NN
token='7928522469:AAHDVmQ9qpxQpgAtrNkf2c62SWOHxjq03xY'
bot = telebot.TeleBot(token)
chek = False
@bot.message_handler(commands=['start'])
def start_message(message):
    with open('NN.py', 'rb') as file:
        bot.send_document(message.chat.id, file)
#   bot.send_message(message.chat.id,'')
  

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    global chek
    if "check" in message.text:
        chek = True
        bot.send_message(message.chat.id,'пришлите фото для проверки')
        
  

    

def add_to_csv(name, comp):
    with open("compic.csv", 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([name, comp])

SAVE_DIR = 'images'
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists("compic.csv"):
    with open("compic.csv", 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['name', 'comp'])

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    global chek
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image = Image.open(BytesIO(downloaded_file)).convert('RGB')
    random_uuid = uuid.uuid4()
    filename = f"{random_uuid}.png"
    image.save(os.path.join(SAVE_DIR, filename))
    if not chek:
        
        
        markup = types.InlineKeyboardMarkup()
        btn1 = types.InlineKeyboardButton("Компактный", callback_data=f"1 {filename}")
        btn2 = types.InlineKeyboardButton("Что-то среднее", callback_data=f"0 {filename}")
        btn3 = types.InlineKeyboardButton("Некомпактный", callback_data=f"-1 {filename}")
        markup.add(btn1, btn2, btn3)
        bot.send_message(message.chat.id, "Выбери компактность данного животного:", reply_markup=markup)
    else:
        bot.send_message(message.chat.id, NN.start(f"images/{filename}"))
        # (NN.start(f"images/{filename}"))
        chek = False

@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    value, filename = call.data.split()
    print(f"💾 Сохраняем: {filename}, метка: {(value)}")
    add_to_csv(filename, int(value))
    dic = {
        "1":"Компактный",
        "0":"Что-то среднее",
        "-1":"Некомпактный"
    }
    bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"ваш ответ: {dic[value]}",
            reply_markup=None  # Убираем клавиатуру
        )
    bot.answer_callback_query(call.id, "Спасибо, ответ записан!")

    
    # bot.reply_to(message, f"📸 Фото сохранено как {filename}")
    # bot.reply_to(message, "оцените компактность данного животного от 0 до 1, где 0 - совсем не компактный, а 1 - суперкомпактный")

# @bot.message_handler(content_types=['document'])
# def handle_document(message):
#     file_info = bot.get_file(message.document.file_id)
#     downloaded_file = bot.download_file(file_info.file_path)

#     image = Image.open(BytesIO(downloaded_file)).convert('RGB')
#     random_uuid = uuid.uuid4()
#     filename = f"{random_uuid}.png"
#     image.save(os.path.join(SAVE_DIR, filename))
#     markup = types.InlineKeyboardMarkup()
#     btn1 = types.InlineKeyboardButton("Компактный", callback_data=f"1 {filename}")
#     btn2 = types.InlineKeyboardButton("Что-то среднее", callback_data=f"0 {filename}")
#     btn3 = types.InlineKeyboardButton("Некомпактный", callback_data=f"-1 {filename}")
#     markup.add(btn1, btn2, btn3)
#     bot.send_message(message.chat.id, "Выбери компактность данного животного:", reply_markup=markup)

#     # bot.reply_to(message, f"📄 Документ сохранён как {filename}")
    

    

    
bot.polling(none_stop=True)
