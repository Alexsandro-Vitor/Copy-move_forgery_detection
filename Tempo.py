import time

tempo = time.time()
i = 0
while i < 1000000:
	i += 1
print(time.time() - tempo)