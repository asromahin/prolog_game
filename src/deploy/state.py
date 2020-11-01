class GlobalState:
    def __init__(self):
        self.last_query = ''
        self.last_image = None

    def add_query(self, query):
        self.last_query = query

    def add_image(self, image):
        self.last_image = image