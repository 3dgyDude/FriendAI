import os
import re
import pandas as pd

from random import randrange

pattern = r'([a-zA-Z\s]+):(.+)'
data = {
    'name': [],
    'line': []
}
filename = 'javi.txt' # Cambiar por el archivo de las conversaciones con tu amigo
friendname = ' Javivivivu' # Nombre del amigo en el archivo de whatsapp

# Prepara el dataset
with open('javi.txt', encoding="utf8") as file:
  for line in file.readlines():
    match = re.findall(pattern, line)
    if match:
      name, line = match[0]
      if line != ' <Multimedia omitido>':
        if len(data['name']) > 0 and name == data['name'][len(data['name']) - 1]:
          data['line'][len(data['name']) - 1] = data['line'][len(data['name']) - 1] + '\n' + line
        else:
          data['name'].append(name)
          data['line'].append(line)

# Muestra información útil 
df = pd.DataFrame(data)
print("\nTamaño dataset: ", len(df))
print("Ejemplos:\n")
print(df.sample(10))

print("\nLíneas totales del amigo = ", sum(df['name'] == friendname), "\n")

# Exporta todo a csv
df.to_csv('javi.csv', index=False)