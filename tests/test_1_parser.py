# test_1_parser.py

from src.parser import AcademicParser

# Aponta para a pasta com apenas 1 ou 2 arquivos
parser = AcademicParser("./data")
docs = parser.process_all()

# Verifica o resultado
for i, doc in enumerate(docs[:3]): # Mostra os 3 primeiros trechos
    print(f"\n--- Documento {i+1} ---")
    print(f"Metadados: {doc.metadata}")
    print(f"Conteúdo (primeiros 100 caracteres): {doc.page_content[:100]}...")