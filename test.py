#!/usr/bin/env python
"""
@author: phdenzel

The entire test suite
"""
from tests.prototype import SequentialTestLoader
from tests.scryfallbase_test import ScryfallBaseTest
from tests.scryfallcardname_test import ScryfallCardNameTest
from tests.scryfallsearch_test import ScryfallSearchTest
from tests.scryfallsets_test import ScryfallSetsTest
from tests.card_test import CardTest
from tests.sqlcommands_test import SQLCommandsTest



loader = SequentialTestLoader()
# loader.proto_load(ScryfallBaseTest)
# loader.proto_load(ScryfallCardNameTest)
# loader.proto_load(ScryfallSearchTest)
# loader.proto_load(ScryfallSetsTest)
loader.proto_load(CardTest)
# loader.proto_load(SQLCommandsTest)

loader.run_suites(verbosity=1)
