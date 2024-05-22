from dataclasses import dataclass
@dataclass
class synonym_item:
    """Class for keeping track of a synonym item in retrofitting."""
    word: str
    time: str
    speaker: str
    party: str
    debate: str = None

    def stringify(self) -> str:
        valid_parts = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                valid_parts.append(value)
        return "$".join(valid_parts)

    @classmethod
    def from_string(cls, input_string):
        i = input_string.split('$')
        assert len(i) >= 4
        return cls(*i)