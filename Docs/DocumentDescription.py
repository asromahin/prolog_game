
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
                name='department',
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
    DocumentDescription(
        name='agreed_indicators',
        rus_name='Согласованные технико-экономические показатели по объекту',
        fields=[
            DocumentField(
                name='square_building',
                rus_name='Площадь застройки',
            ),
            DocumentField(
                name='volume',
                rus_name='Объём',
            ),
            DocumentField(
                name='number_storeys',
                rus_name='Этажность',
            ),
            DocumentField(
                name='top_indicator',
                rus_name='Верхняя отметка объекта',
            ),
            DocumentField(
                name='full_square',
                rus_name='Общая площадь',
            ),
            DocumentField(
                name='full_square_with_on_ground',
                rus_name='В том числе надземная',
            ),
            DocumentField(
                name='on_ground_square',
                rus_name='Надземная',
            ),
        ]
    ),
    DocumentDescription(
        name=' building_permission_capital',
        rus_name='Разрешение на ввод Объекта капитального строительства',
        fields=[
            DocumentField(
                name='address',
                rus_name='Расположенного по адресу',
            ),
            DocumentField(
                name='building_address',
                rus_name='Строительный адрес',
            ),
            DocumentField(
                name='volume_proj_fact',
                rus_name='Строительный объем – всего (по проекту, фактически)',
            ),
            DocumentField(
                name='volume_proj_fact_include_on_ground',
                rus_name='В том числе надземной части (по проекту, фактически)',
            ),
            DocumentField(
                name='full_square',
                rus_name='Общая площадь - (по проекту, фактически)',
            ),
            DocumentField(
                name='square_buildings_proj_fact',
                rus_name='Площадь встроенно-пристроенных помещений (по проекту,фактически)',
            ),
            DocumentField(
                name='count_buildings',
                rus_name='Количество зданий (шт)',
            ),
        ]
    ),
    DocumentDescription(
        name=' building_permission_capital',
        rus_name='Разрешение на ввод Объекта капитального строительства',
        fields=[
            DocumentField(
                name='objects_not_industry',
                rus_name='Объекты непроизводственного назначения',
            ),
            DocumentField(
                name='number_proj_fact',
                rus_name='Количество мест (по проекту, фактически)',
            ),
            DocumentField(
                name='number_visits_proj_fact',
                rus_name='Количество посещений (по проекту, фактически)',
            ),
            DocumentField(
                name='capacity_volume_proj_fact',
                rus_name='Вместимость (по проекту, фактически)',
            ),
            DocumentField(
                name='full_square_parking',
                rus_name='Общая площадь подземной автостоянки (по проекту, фактически)',
            ),
            DocumentField(
                name='square_admin_buildings_proj_fact',
                rus_name='Общая площадь административных помещений (по проекту, фактически)',
            ),
            DocumentField(
                name='square_culture_buildings_proj_fact',
                rus_name='Общая площадь помещений культурно-досугового назначения (по проекту, фактически)',
            ),
            DocumentField(
                name='count_buildings',
                rus_name='Торговая площадь (по проекту, фактически)',
            ),
            DocumentField(
                name='count_buildings',
                rus_name='Этажность (по проекту, фактически)',
            ),
        ]
    ),
]


Объекты непроизводственного назначения
Количество мест (по проекту, фактически)
Количество посещений (по проекту, фактически)
Вместимость (по проекту, фактически)
Общая площадь подземной автостоянки (по проекту, фактически)
Общая площадь административных помещений (по проекту, фактически)
Общая площадь помещений культурно-досугового назначения (по проекту, фактически)
Торговая площадь (по проекту, фактически)
Этажность (по проекту, фактически)