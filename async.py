from MyModule import asynctest
import asyncio

test = asynctest.AsyncTest()

async def process():
    test.connect('127.0.0.1',6000)
    await asyncio.wait([
        asyncio.create_task(test.asyncRepeat()),
       asyncio.create_task(test.write())])

asyncio.run(process())