import logging
import nest_asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext
from telegram.ext.filters import PHOTO
from model_predict_boris import predict_image as predict_boris
from model_predict_nikita import pred_disease as predict_nikita
import asyncio

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = '7643982179:AAEtBUGp0tT94IkqOdhN6hk-gMUl_zBhYk8'

current_model = "boris"

def predict_image_with_model(image_bytes):
    if current_model == "boris":
        return predict_boris(image_bytes)
    elif current_model == "nikita":
        return predict_nikita(image_bytes)
    else:
        return "Ошибка: модель не выбрана."


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Привет! Отправь мне изображение, и я сделаю предсказание. "
        "Также ты можешь выбрать модель для предсказания с помощью команд:\n"
        "/boris - модель Бориса\n"
        "/nikita - модель Никиты"
    )


async def set_model_boris(update: Update, context: CallbackContext) -> None:
    global current_model
    current_model = "boris"
    await update.message.reply_text("Модель переключена на Бориса!")


async def set_model_nikita(update: Update, context: CallbackContext) -> None:
    global current_model
    current_model = "nikita"
    await update.message.reply_text("Модель переключена на Никиту!")


async def handle_photo(update: Update, context: CallbackContext) -> None:
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_bytes = await file.download_as_bytearray()

    prediction = predict_image_with_model(image_bytes)
    await update.message.reply_text(f"Предсказание: {prediction}")


# Основной цикл бота
async def main() -> None:
    application = Application.builder().token(API_TOKEN).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("boris", set_model_boris))
    application.add_handler(CommandHandler("nikita", set_model_nikita))

    application.add_handler(MessageHandler(PHOTO, handle_photo))

    await application.run_polling()


if __name__ == '__main__':
    nest_asyncio.apply()
    asyncio.run(main())
