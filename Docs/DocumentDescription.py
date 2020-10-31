
class DocumentDescription:
    def __init__(self, name, rus_name, fields):
        self.rus_name = rus_name
        self.name = name
        doc_fields = [
            DocumentField(
                name='type_doc',
                rus_name='Тип документа',
            ),
            DocumentField(
                name='number_doc',
                rus_name='Номер документа',
            ),
            DocumentField(
                name='issue_date',
                rus_name='Дата документа',
            ),
            DocumentField(
                name='authority',
                rus_name='Выдавший орган',
            ),
        ]
        self.fields = doc_fields + fields


class DocumentField:
    def __init__(self, name, rus_name):
        self.name = name
        self.rus_name = rus_name


list_of_documents = [
    DocumentDescription(
        name='certificate_solution',
        rus_name='Свидетельство об утверждении архитектурно-градостроительного решения',
        fields=[
            DocumentField(
                name='',
                rus_name='Административный округ',
            ),
            DocumentField(
                name='district',
                rus_name='Район',
            ),
            DocumentField(
                name='address',
                rus_name='Адресс',
            ),
            DocumentField(
                name='name_obj',
                rus_name='Наименование объекта',
            ),
            DocumentField(
                name='function_of_object',
                rus_name='Функциональное назначение объекта',
            ),
        ]
    ),
]


