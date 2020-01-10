from mtg.types import conditions as cond
from mtg.types import languages as lang


class SQLString(str):
    def __new__(cls, keys=None, types=None):
        if isinstance(keys, (tuple, list)):
            types = ['TEXT']*len(keys) if not types else types
            value = ", ".join(["{} {}".format(k, t) for k, t in zip(keys, types)])
        else:
            value = keys
            keys = types = ()
        obj = str.__new__(cls, value)
        obj.keys = keys
        obj.types = types
        return obj


ARKHIVE_keys = ("card_id", "amount", "name", "edition", "collector_number",
                "condition", "language", "details", "type")
ARKHIVE_types = ("INTEGER", "INTEGER", "TEXT", "TEXT", "INTEGER PRIMARY KEY",
                 "INTEGER DEFAULT {}".format(cond.mint),
                 "TEXT DEFAULT '{}'".format(lang.english), "TEXT", "TEXT")
ARKHIVE = SQLString(ARKHIVE_keys, ARKHIVE_types)


__all__ = {'ARKHIVE': ARKHIVE}
