import re
from pathlib import Path
from langchain_core.documents import Document

class AcademicParser:
    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)
        # Seus headers específicos na ordem do seu template
        self.headers = [
            "TITLE", "AUTHOR", "JOURNAL", "ABSTRACT", "INTRO", 
            "LITERATURE", "METHODOLOGY", "RESULTS_AND_DISCUSSION", 
            "IMPLICATION_LIMITATION_FUTURERESEARCH", "CONCLUSION", 
            "EXTRAS", "REFERENCES"
        ]
        # Regex que procura por "# NOME_DO_HEADER" no início de cada linha
        self.header_pattern = re.compile(r'^#\s+(' + '|'.join(self.headers) + r')', re.MULTILINE)

    def parse_file(self, file_path):
        """Transforma um arquivo .md em uma lista de Documents do LangChain."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Divide o conteúdo usando o padrão dos headers
        # O split gera uma lista: ['', 'HEADER1', 'TEXTO1', 'HEADER2', 'TEXTO2'...]
        parts = self.header_pattern.split(content)
        
        # 2. Organiza em um dicionário para fácil acesso
        data = {}
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i+1].strip()
            # Remove o separador '---' se ele sobrar no corpo do texto
            body = body.replace('---', '').strip()
            data[header] = body

        # 3. Extrai Metadados Globais (para busca filtrada)
        global_metadata = {
            "source": file_path.name,
            "title": data.get("TITLE", "N/A").split('\n')[0].strip(),
            "author": data.get("AUTHOR", "N/A").split('\n')[0].strip(),
            "journal": data.get("JOURNAL", "N/A").split('\n')[0].strip()
        }

        documents = []
        # Seções que realmente queremos que a IA "leia" para responder
        sections_to_index = [
            "ABSTRACT", "INTRO", "LITERATURE", "METHODOLOGY", 
            "RESULTS_AND_DISCUSSION", "IMPLICATION_LIMITATION_FUTURERESEARCH", "CONCLUSION"
        ]
        
        for section in sections_to_index:
            text = data.get(section, "").strip()
            
            # Só indexamos se houver conteúdo real (ignora "EMPTY" ou seções curtas demais)
            if text and text.upper() != "EMPTY" and len(text) > 40:
                meta = global_metadata.copy()
                meta["section"] = section
                
                # Contextualização: Injetamos o título e autor no próprio texto do chunk.
                # Isso ajuda o modelo de embedding (BGE-M3) a associar o conteúdo ao contexto correto.
                content_with_context = (
                    f"Article Title: {meta['title']}\n"
                    f"Author: {meta['author']}\n"
                    f"Section: {section}\n"
                    f"Content: {text}"
                )
                
                documents.append(Document(page_content=content_with_context, metadata=meta))
        
        return documents

    def process_all(self):
        """Varre a pasta e processa todos os arquivos .md encontrados."""
        all_docs = []
        files = list(self.directory_path.glob("*.md"))
        
        if not files:
            print(f"⚠️ Aviso: Nenhum arquivo .md encontrado em {self.directory_path}")
            return []

        for f in files:
            try:
                all_docs.extend(self.parse_file(f))
            except Exception as e:
                print(f"❌ Erro ao processar {f.name}: {e}")
                
        return all_docs