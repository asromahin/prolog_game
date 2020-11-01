from deeppavlov import build_model, configs


class DeepPavlovPipeline:
    def __init__(self):
        self.model = build_model(configs.squad.squad_ru_bert_infer, download=True)

    def predict(self, text, query='расположенного по адресу'):
        return self.model([text], [query])

