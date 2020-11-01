
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
    def __init__(self, name, rus_name, proj_fact=False):
        self.name = name
        self.rus_name = rus_name
        self.proj_fact = proj_fact


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
                rus_name='Адрес',
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
                name='building_volume',
                rus_name='Строительный объем – всего (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='volume_include_on_ground',
                rus_name='В том числе надземной части (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='full_square',
                rus_name='Общая площадь - (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='square_buildings',
                rus_name='Площадь встроенно-пристроенных помещений (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='count_buildings',
                rus_name='Количество зданий (шт)',
            ),
        ]
    ),
    DocumentDescription(
        name='building_permission_capital',
        rus_name='Разрешение на ввод Объекта капитального строительства',
        fields=[
            DocumentField(
                name='objects_not_industry',
                rus_name='Объекты непроизводственного назначения',
            ),
            DocumentField(
                name='number_places',
                rus_name='Количество мест (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='number_visits',
                rus_name='Количество посещений (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='capacity',
                rus_name='Вместимость (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='full_square_parking',
                rus_name='Общая площадь подземной автостоянки (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='square_admin_buildings',
                rus_name='Общая площадь административных помещений (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='square_culture_buildings',
                rus_name='Общая площадь помещений культурно-досугового назначения (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='count_buildings',
                rus_name='Торговая площадь (по проекту, фактически)',
                proj_fact=True,
            ),
            DocumentField(
                name='level_buildings',
                rus_name='Этажность (по проекту, фактически)',
                proj_fact=True,
            ),
        ]
    ),
DocumentDescription(
        name='town_planning_conclusion',
        rus_name='Градостроительное заключение',
        fields=[
            DocumentField(
                name='doc_date',
                rus_name='Дата документа',
            ),
            DocumentField(
                name='doc_number',
                rus_name='Номер документа',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='land_area',
                rus_name='Площадь участка',
            ),
        ]
    ),
DocumentDescription(
        name='completed_object_acceptance',
        rus_name='Акт государственной приемочной комиссии о приемке в эксплуатацию законченного строительством объекта',
        fields=[
            DocumentField(
                name='doc_date',
                rus_name='Дата документа',
            ),
            DocumentField(
                name='doc_number',
                rus_name='Номер документа',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='land_area',
                rus_name='Площадь участка (по проекту и фактически)',
            ),
            DocumentField(
                name='work_type',
                rus_name='Тип работ',
            ),
            DocumentField(
                name='work_indicators',
                rus_name='Показатели работ (по проекту и фактически)',
            ),
            DocumentField(
                name='storeys_number',
                rus_name='Этажность (по проекту и фактически)',
            ),
            DocumentField(
                name='sections_number',
                rus_name='Количество секций (по проекту и фактически)',
            ),
            DocumentField(
                name='construction_volume',
                rus_name='Строительный объем (по проекту и фактически)',
            ),
        ]
    ),
DocumentDescription(
        name='floor_plan_legend',
        rus_name='Экспликация к архивному поэтажному плану',
        fields=[
            DocumentField(
                name='storey',
                rus_name='Этаж',
            ),
            DocumentField(
                name='doc_number',
                rus_name='Номер документа',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='land_area',
                rus_name='Площадь участка',
            ),
            DocumentField(
                name='premises',
                rus_name='Помещение',
            ),
            DocumentField(
                name='total_floor_area',
                rus_name='Общая площадь этажа',
            ),
            DocumentField(
                name='total_building_area',
                rus_name='Общая площадь здания',
            ),
            DocumentField(
                name='altitude',
                rus_name='Высотность',
            ),
            DocumentField(
                name='plan_date',
                rus_name='Дата составления плана',
            ),
        ]
    ),
DocumentDescription(
        name='building_permit',
        rus_name='Разрешение на строительство',
        fields=[
            DocumentField(
                name='height',
                rus_name='Высота',
            ),
            DocumentField(
                name='building_address',
                rus_name='Строительный адрес',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='total_area',
                rus_name='Общая площадь',
            ),
            DocumentField(
                name='doc_validity',
                rus_name='Срок действия документа',
            ),
            DocumentField(
                name='work_indicators',
                rus_name='Показатели работ (по проекту и фактически)',
            ),
            DocumentField(
                name='storeys_number',
                rus_name='Количество этажей',
            ),
            DocumentField(
                name='underground_storeys_number',
                rus_name='Подземные этажи',
            ),
            DocumentField(
                name='volume',
                rus_name='Объем',
            ),
            DocumentField(
                name='built_up_area',
                rus_name='Площадь застройки',
            ),
            DocumentField(
                name='land_area',
                rus_name='Площадь участка',
            ),
        ]
    ),
DocumentDescription(
        name='permit_for_construction_works',
        rus_name='Разрешение на производство строительно-монтажных работ',
        fields=[
            DocumentField(
                name='administrative_district',
                rus_name='Административный округ',
            ),
            DocumentField(
                name='moscow_distict',
                rus_name='Район города Москвы',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='build_type',
                rus_name='Вид строительства',
            ),
            DocumentField(
                name='doc_validity',
                rus_name='Срок действия документа',
            ),
        ]
    ),
DocumentDescription(
        name='prefect_order',
        rus_name='Распоряжение префекта',
        fields=[
            DocumentField(
                name='doc_date',
                rus_name='Дата документа',
            ),
            DocumentField(
                name='doc_number',
                rus_name='Номер документа',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='land_area',
                rus_name='Площадь участка',
            ),
        ]
    ),
DocumentDescription(
        name='permit_for_preparatory_basic_construction_works',
        rus_name='Разрешение на производство подготовительных и основных строительно-монтажных работ',
        fields=[
            DocumentField(
                name='administrative_district',
                rus_name='Административный округ',
            ),
            DocumentField(
                name='moscow_district',
                rus_name='Район города Москвы',
            ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='build_type',
                rus_name='Вид строительства',
            ),
            DocumentField(
                name='doc_validity',
                rus_name='Срок действия документа',
            ),
        ]
    ),
DocumentDescription(
        name='lease_contract',
        rus_name='Договор аренды земного участка',
         fields=[
            DocumentField(
                name='doc_date',
                rus_name='Дата документа',
             ),
            DocumentField(
                name='doc_number',
                rus_name='Номер документа',
                ),
            DocumentField(
                name='doc_validity',
                rus_name='Срок действия документа',
            ),
        ]
    ),
DocumentDescription(
        name='tech_passport',
        rus_name='Технический паспорт',
         fields=[
            DocumentField(
                name='purpose_object',
                rus_name='Назначение объекта',
             ),
            DocumentField(
                name='address',
                rus_name='Адрес',
            ),
            DocumentField(
                name='owner',
                rus_name='Владелец',
            ),
             DocumentField(
                 name='year',
                 rus_name='Год постройки',
             ),
             DocumentField(
                 name='floor',
                 rus_name='Этаж',
             ),
             DocumentField(
                 name='roof_area',
                 rus_name='Площадь крыши',
             ),
             DocumentField(
                 name='volume',
                 rus_name='Объем',
             ),
             DocumentField(
                 name='total_area',
                 rus_name='Общая площадь',
             ),
        ]
    ),

]