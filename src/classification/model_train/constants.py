DOCNAME2CLASS = {'БТИ' : 0,
                 'ЗУ' : 1,
                 'Разр. на ввод' : 2,
                 'Разр. на стр-во': 3,
                 'Свид. АГР': 4}


BACKDOCNAME2CLASS = {idx:key for key, idx in list(zip(DOCNAME2CLASS.keys(), DOCNAME2CLASS.values()))}