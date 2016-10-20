import collections


class NutszebraDictionary(dict):
    """Adapted dictionary

    Note:
       | objects which contains __iter__ method, such as yielder and list and tuple, can be the key of this dict.
       | However, builtin dict is not considered to be the key of this dict, even though builtin dict contains __iter__ method

    Example:

        ::

           # tuple case
           test = nutszebra_basic_dictionary.NutszebraDictionary()
           test[(1, 2, 3)] = 'hi'
           >>> print(test)
               {1: {2: {3: 'hi'}}}
           >>> print(test[(1, 2, 3)])
               'hi'

           # list case
           test = nutszebra_basic_dictionary.NutszebraDictionary()
           test[[1, 2, 3]] = 'hi'
           >>> print(test)
               {1: {2: {3: 'hi'}}}
           >>> print(test[[1, 2, 3]])
               'hi'

           # yielder case
           test = nutszebra_basic_dictionary.NutszebraDictionary()
           test[six.moves.range(1, 4)] = 'hi'
           >>> print(test)
               {1: {2: {3: 'hi'}}}
           >>> print(test[six.moves.range(1, 4)])
               'hi'

           # insert case
           test = nutszebra_basic_dictionary.NutszebraDictionary({1: {2: {3: 'hi'}}})
           >>> print(test)
               {1: {2: {3: 'hi'}}}
           >>> print(test[six.moves.range(1, 4)])
               'hi'

    Attributes:
    """

    def __getitem__(self, key, dict_getitem=dict.__getitem__):
        """Get value via object with __iter__

        Edited date:
            160628

        Test:
            160628
        """
        # iterable keys are allowed
        if type(key) is not str and isinstance(key, collections.Iterable):
            val = self
            for i in key:
                # when converting json, the key of number is converted into str compulsory, thus keys are converted as str here
                i = str(i)
                val = dict_getitem(val, i)
        else:
            key = str(key)
            val = dict_getitem(self, key)
        return val

    def __setitem__(self, key, val, dict_getitem=dict.__getitem__,  dict_setitem=dict.__setitem__):
        """Set value via object with __iter__

        Edited date:
            160628

        Test:
            160628
        """
        # iterable keys are allowed
        tmp = self
        if type(key) is not str and isinstance(key, collections.Iterable):
            # error flag
            flag = True
            for i in key[:-1]:
                # when converting json, the key of number is converted into str compulsory, thus keys are converted as str here
                i = str(i)
                try:
                    if flag:
                        # no errors  occur yet
                        tmp = dict_getitem(tmp, i)
                    else:
                        # error happened
                        dict_setitem(tmp, i, {})
                        tmp = dict_getitem(tmp, i)
                except:
                    # error flag
                    flag = False
                    # set dictionary
                    dict_setitem(tmp, i, {})
                    tmp = dict_getitem(tmp, i)
            # set key
            # when converting json, the key of number is converted into str compulsory, thus keys are converted as str here
            key = key[-1]
        # set value
        dict_setitem(tmp, str(key), val)
