"""
© 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.
"""
from numbers import Number
from inspect import signature, Parameter
from numpy import ndarray, array, arange


# # Subsets stored within a dict that is itself a single class attribute
# class ParameterSet():
#     """
#     A data type that maps from a full parameter set, by means of a subset
#     name (str), to the parameter values that represent the specified subset.
#     It allows setting & getting of values even when the subset used to set
#     the values is different from the subset use to get the values.
#     """
#     def __init__(self, **kwargs):
#         self.subset_ind = {}
#         val_list, n = self.align_indicies(kwargs, [], 0)
#         self.full_set = array(val_list)

#     def align_indicies(self, kwargs, val_list, n):
#         for param, val in kwargs.items():
#             if isinstance(val, Number):
#                 l = 1
#                 val_list.append(val)
#                 self.subset_ind[param] = n
#             elif isinstance(val, ndarray):
#                 l = val.size
#                 val_list.extend(val.reshape(-1))
#                 self.subset_ind[param] = arange(n, n + l)
#             elif isinstance(val, dict):
#                 n0 = n
#                 val_list, n1 = self.align_indicies(val, val_list, n0)
#                 l = n1 - n0
#                 self.subset_ind[param] = arange(n, n + l)
#             n += l
#         return val_list, n
    
#     @staticmethod
#     def parse_slicer(name):
#         if '[' in name:
#             slice_str = name[(name.index('[') + 1):name.index(']')]
#             if ':' not in slice_str:
#                 slicer = int(slice_str)
#             else:
#                 slicer = slice(*[int(i) if i else None
#                                  for i in slice_str.split(':')])
#             name = name[:name.index('[')]
#         else:
#             slicer = slice(None)
#         return name, slicer

#     def set(self, subset, val):
#         subset, slicer = self.__class__.parse_slicer(subset)
#         subset_ind = self.subset_ind[subset]
#         if isinstance(subset_ind, Number):
#             self.full_set[subset_ind] = val
#         elif isinstance(subset_ind, ndarray):
#             self.full_set[subset_ind[slicer]] = val

#     def get(self, subset):
#         subset, slicer = self.__class__.parse_slicer(subset)
#         subset_ind = self.subset_ind[subset]
#         if isinstance(subset_ind, Number):
#             return self.full_set[subset_ind]
#         elif isinstance(subset_ind, ndarray):
#             return self.full_set[subset_ind][slicer]

#     def def_subset(self, name, params):
#         ind = []
#         for param in params:
#             param, slicer = self.__class__.parse_slicer(param)
#             param_ind = self.subset_ind[param]
#             if isinstance(param_ind, Number):
#                 ind.append(param_ind)
#             elif isinstance(param_ind, ndarray):
#                 ind.extend(param_ind[slicer])
#         self.subset_ind[name] = array(ind, dtype='uint')

# my_ps = ParameterSet(a=1, b=array([2, 3]), c=dict(d=4, e=array([5, 6, 7, 8])))
# print(my_ps.full_set)
# print(my_ps.subset_ind)
# print(my_ps.get('e[1:3]'))
# my_ps.set('a', 10)
# my_ps.set('e[1:3]', array([60, 70]))
# print(my_ps.full_set)
# print(my_ps.get('a'))
# my_ps.def_subset('active', ['a', 'e[2:4]'])
# print(my_ps.subset_ind)
# my_ps.set('active', array([100, 500, 600]))
# print(my_ps.full_set)
# print(my_ps.get('active'))
# print(my_ps.get('a'))


# Subsets each stored as their own class attribute
class ParameterSet():
    """
    A data type that maps from a full parameter set, by means of a subset
    name (str), to the parameter values that represent the specified subset.
    It allows setting & getting of values even when the subset used to set
    the values is different from the subset use to get the values.
    Alternatively think of this as a name space with multiple mappings to
    individual data members.
    """
    # TODO: Is it worth making this a proper metaclass?
    # TODO: Currently, when getting a subset this only provides the corresponding array.
    #       There is a need for getting a subset as a dictionary. Would that be possible?
    #       Yes, could create a method that is a variation on set_default_params below?
    def __new__(cls, **kwargs):
        obj = super().__new__(cls)
        val_list, n = cls.align_indicies(obj, kwargs, [], 0)
        obj.full_set = array(val_list)
        return obj

    @classmethod
    def property_factory(cls, obj, prop_name, subset_ind):
        setattr(obj, '_' + prop_name, subset_ind)
        getter = lambda self, name=('_' + prop_name): self.generic_get(name)
        setter = (lambda self, value, name=('_' + prop_name):
                  self.generic_set(name, value))
        setattr(cls, prop_name, property(fget=getter, fset=setter))

    @classmethod
    def align_indicies(cls, obj, kwargs, val_list, n):
        for param, val in kwargs.items():
            if isinstance(val, Number):
                l = 1
                val_list.append(val)
                subset_ind = n
            elif isinstance(val, ndarray):
                l = val.size
                val_list.extend(val.reshape(-1))
                subset_ind = arange(n, n + l)
            elif isinstance(val, dict):
                n0 = n
                val_list, n1 = cls.align_indicies(obj, val, val_list, n0)
                l = n1 - n0
                subset_ind = arange(n, n + l)
            cls.property_factory(obj, param, subset_ind)
            n += l
        return val_list, n
    
    @staticmethod
    def parse_slicer(name):
        if '[' in name:
            slice_str = name[(name.index('[') + 1):name.index(']')]
            if ':' not in slice_str:
                slicer = int(slice_str)
            else:
                slicer = slice(*[int(i) if i else None
                                 for i in slice_str.split(':')])
            name = name[:name.index('[')]
        else:
            slicer = slice(None)
        return name, slicer

    def generic_set(self, subset_private, val):
        subset_ind = getattr(self, subset_private)
        self.full_set[subset_ind] = val

    def generic_get(self, subset_private):
        subset_ind = getattr(self, subset_private)
        return self.full_set[subset_ind]

    def def_subset(self, name, params):
        ind = []
        for param_full in params:
            param, slicer = self.__class__.parse_slicer(param_full)
            si = getattr(self, '_' + param)  # subset_index
            param_ind = si[slicer] if isinstance(si, ndarray) else si
            if isinstance(param_ind, Number):
                ind.append(param_ind)
            elif isinstance(param_ind, ndarray):
                ind.extend(param_ind)
        subset_ind = array(ind, dtype='uint')
        self.__class__.property_factory(self, name, subset_ind)


def get_default_params(my_func):
    """
    Create a dictionary of the default arguments for the provided function,
    my_func, following the advice on stackoverflow:
    https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    my_sig = signature(my_func)
    return {param : val.default for param, val in my_sig.parameters.items()
            if val.default is not Parameter.empty}

def set_default_params(my_func, my_parameter_set):
    """
    Create a dictionary that pulls values from my_parameter_set according the the names specified
    as default parameters of my_func.
    """
    my_sig = signature(my_func)
    return {param : getattr(my_parameter_set, param) for param, val in my_sig.parameters.items()
            if val.default is not Parameter.empty}

if __name__ == "__main__":
    # # With only one ParameterSet object (saves data into the class definition):
    # ps = ParameterSet(a=1, b=array([2, 3]), c=dict(d=4, e=array([5, 6, 7, 8])))
    # When multiple instances from ParameterSet are needed (dynamic subclass):
    ps = type('Sub', (ParameterSet,), {})(a=1, b=array([2, 3]),
                                          c=dict(d=4, e=array([5, 6, 7, 8])))
    print(ps.full_set)
    ps.a = 10  # set a value
    print(ps.full_set)
    print(ps.a)  # get a value
    ps.def_subset('active', ['a', 'b[1]', 'e[2:4]'])  # add subsets
    ps.active = array([100, 300, 700, 800])  # set multiple values
    print(ps.full_set)
    print(ps.active)  # get multiple values
    print(ps.a)  # demonstrate that setting one subset affects other subsets

    # Visualize where the information is stored:
    print(' ')
    print('ParameterSet — name & dict:')
    print(ParameterSet.__name__)
    print(ParameterSet.__dict__)
    print(' ')
    print('ps.__class__ — name & dict:')
    print(ps.__class__.__name__)
    print(ps.__class__.__dict__)
    print(' ')
    print('ps — dict:')
    print(ps.__dict__)