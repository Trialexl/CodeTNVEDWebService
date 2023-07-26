import requests
from aiogram import Bot, Dispatcher,types, executor
from aiogram.dispatcher.filters.state import State,StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from config import TokenAPI,siteURL
from urllib.parse import urlparse, parse_qs,quote
import json
from datetime import datetime

bot = Bot(token=TokenAPI)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)



      
@dp.message_handler()
async def GetGroupCode(message: types.Message, state: FSMContext):
    
    text = message.text
    data ={
            "text": text,
            "qty": 3
        }
    response = requests.get(siteURL, params=data)
    
    json_data = response.json()


if __name__=='__main__':
    executor.start_polling(dp)