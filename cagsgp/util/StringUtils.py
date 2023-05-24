class StringUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def only_first_char_upper(s: str) -> str:
        return s[0].upper() + s[1:]
    