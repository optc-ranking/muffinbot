import os
import asyncio
import datetime
import discord
from google import genai
from google.genai import types

TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = 'gemini-2.5-flash'
client = genai.Client(api_key=GEMINI_API_KEY)

DEFAULT_BUDGET = 128
CONTEXT_HOURS = 24

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

async def collect_context(channel, current_message):
    """Gather messages from the last CONTEXT_HOURS in the channel."""
    after_time = datetime.datetime.utcnow() - datetime.timedelta(hours=CONTEXT_HOURS)
    context_parts = []
    async for msg in channel.history(limit=None, after=after_time, oldest_first=True):
        if msg.id == current_message.id:
            continue
        context_parts.append(f"{msg.author.display_name}: {msg.clean_content}")
    return "\n".join(context_parts)

@bot.event
async def on_message(message):
    if bot.user.mentioned_in(message) and message.author != bot.user:
        prompt = message.content
        prompt = prompt.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        thinking = False
        if '!think' in prompt:
            thinking = True
            prompt = prompt.replace('!think', '').strip()
        context_text = await collect_context(message.channel, message)
        full_prompt = (context_text + '\n' if context_text else '') + f"{message.author.display_name}: {prompt}"
        config = None
        if thinking:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=DEFAULT_BUDGET)
            )
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=config,
        )
        await message.channel.send(response.text)

bot.run(TOKEN)
