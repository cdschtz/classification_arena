from torchtext import data

from utils import get_files_with_data


class WikiSyntheticGeneral(data.Dataset):

    name = 'WikiSyntheticGeneral'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, **kwargs):
        """Create some stuff."""
        import json

        extraction_fields = {
            "text": [("text", text_field)],
            "label": [("label", label_field)]
        }

        examples = []
        for file_name in get_files_with_data():
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    examples.append(data.Example.fromdict(entry, extraction_fields))

        fields = {
            "text": text_field,
            "label": label_field
        }

        super(WikiSyntheticGeneral, self).__init__(examples, fields, **kwargs)
