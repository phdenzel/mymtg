import numpy as np
from typing import List
from datetime import datetime
from mtg.scryfall import ScryfallCardName, ScryfallSearch
import mtg.types.conditions as cond
import mtg.types.languages as lang
import mtg.types.archetypes as atype
from mtg.types.tables import ARKHIVE


class BaseObject(object):
    def __init__(self, date_format="%Y-%m-%d", **params):
        fields = [key for key in self.__annotations__]
        for field in fields:
            api_field = field
            if not isinstance(self.__annotations__[field], type):
                setattr(self, field, self.__annotations__[field])
            elif api_field in params:
                try:
                    if self.__annotations__[field] == datetime:
                        attr = datetime.strptime(params.get(api_field), date_format).date()
                    else:
                        attr = self.__annotations__[field](params.get(api_field))
                except:
                    attr = params.get(api_field)
                setattr(self, field, attr)
            else:
                setattr(self, field, None)


class Card(BaseObject):
    # all measurements in units of [mm]
    thickness = 0.305
    width = 63.5
    height = 88.9
    aspect = width / height
    dimensions = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                          dtype=np.float32)
    artbox_width = 59
    artbox_height = 45
    # artbox_dimensions = np.array([])
    textbox_width = 59
    textbox_height = 32.5
    # textbox_dimensions= np.array([])
    namebox_width = None
    namebox_height = None
    # namebox_dimensions = np.array([])
    manabox_width = None
    manabox_height = None
    manabox_dimensions = np.array([])
    set_symbol_dimensions = np.array([])
    border_width = 3
    border_radius = 3
    name: str
    rarity: str
    type_line: str
    set: str
    set_name: str
    collector_number: int
    multiverse_ids: list
    lang: str
    promo: bool
    reprint: bool
    reserved: bool
    power: int
    toughness: int
    cmc: int
    mana_cost: str
    loyalty: int
    color_identity: list
    colors: list
    oracle_text: str
    layout: str
    border_color: str
    frame_effects: list
    foil: False
    full_art: bool
    artist: str
    released_at: datetime
    legalities: dict
    prices: dict
    condition: None

    @property
    def __v__(self):
        keys = ['name', 'rarity', 'type', 'legendary', 'subtype',
                'set', 'collector_number']
        if self.type == 'planeswalker':
            keys.append('mana_cost')
            keys.append('loyalty')
        elif self.type == 'creature':
            keys.append('mana_cost')
            keys.append('power')
            keys.append('toughness')
        elif self.type == 'land':
            keys.append('oracle_text')
      
        return "\n".join([k.ljust(20)+"\t{}".format(self.__getattribute__(k))
                          for k in keys])

    def __str__(self):
        return "Card<'{}'#{}>".format(self.name, self.ID)

    def __repr__(self):
        return self.__str__()

    @property
    def multiverse_id(self):
        if isinstance(self.multiverse_ids, list):
            return int(self.multiverse_ids[0])
        elif isinstance(self.multiverse_ids, str):
            return int(self.multiverse_ids)
        else:
            cid = self.collector_number
            sid = int.from_bytes(self.set.encode(), 'little')
            return sid*10**len(str(cid)) + cid

    @property
    def ID(self):
        return self.multiverse_id

    @property
    def archetype(self):
        archetypes = ['Planeswalker', 'Creature', 'Enchantment',
                      'Instant', 'Sorcery', 'Land', 'Artifact']
        for k in archetypes:
            if k in self.type_line:
                return k.lower()

    @property
    def type(self):
        return self.archetype

    @property
    def subtype(self):
        if '-' in self.type_line:
            return self.type_line.split('-')[-1]
        elif '—' in self.type_line:
            return self.type_line.split('—')[-1].strip()
        else:
            return self.type_line

    @property
    def legendary(self):
        return 'Legendary' in self.type_line
    
    def save_image(self, image_name='cache/image.jpg', version='large', **kwargs):
        """
        Save an image of the card
        """
        ScryfallCardName.save_image(self.name, set=self.set, version=version, **kwargs)

    @classmethod
    def from_txt(cls, txtfile, **kwargs):
        """
        Read cards from text file; default line format: <[amount] [name] ([set]) [collector_number]>
        """
        scryings = ScryfallCardName.from_txt(txtfile, **kwargs)
        for lyst in scryings:
            if lyst:
                N = len(lyst)
                scry = lyst[0]
                yield N*[Card(**scry.data)]
            else:
                yield None
