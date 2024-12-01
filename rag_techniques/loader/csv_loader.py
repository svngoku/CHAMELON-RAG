from langchain_community.document_loaders.csv_loader import CSVLoader


class CsvLoader(CSVLoader):
    def __init__(self, file_path: str, source_column: str = None, encoding: str = "utf-8"):
        super().__init__(file_path, source_column, encoding)
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding


    def load(self):
        loader = CSVLoader(file_path=self.file_path, source_column=self.source_column, encoding=self.encoding)
        docs = loader.load_and_split()
        return docs