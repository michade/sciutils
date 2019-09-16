# -*- coding: utf-8 -*-

import inspect
import itertools


class ProperPartial(object):            
    def __init__(self, fun, *args, **kwargs):
        self._fun = fun
        sig = inspect.signature(fun)
        self._signature = sig
        pos, key, var_pos, var_key = self.split_parameters(sig.parameters.values())
        self._pos_params = pos
        self._key_params = key
        self._var_pos_param = var_pos
        self._var_key_param = var_key
        self._bound_args = sig.bind_partial(*args, **kwargs)
        self._partial_signature = self.partial_signature(self._bound_args)
        
    def __call__(self, *args, **kwargs):
        new_bound_args = self._partial_signature.bind(*args, **kwargs)
        final_args, final_kwargs = self.merge_arguments(new_bound_args)
        return self._fun(*final_args, **final_kwargs)

    def __getitem__(self, item):
        return self._bound_args.arguments[item]

    def __contains__(self, item):
        return item in self._bound_args.arguments

    @staticmethod
    def split_parameters(parameters):
        pos = []
        key = []
        var_pos = None
        var_key = None
        p: inspect.Parameter
        for p in parameters:
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos = p
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                var_key = p
            else:
                (pos if p.default == inspect.Parameter.empty else key).append(p)
        return pos, key, var_pos, var_key

    def partial_signature(self, bound_args):
        # print(bound_args)
        ba_set = set(bound_args.arguments)
        new_parameters = [
            p for p in itertools.chain(self._pos_params, self._key_params)
            if p.name not in ba_set
        ]
        if self._var_pos_param is not None:
            new_parameters.append(self._var_pos_param)
        if self._var_key_param is not None:
            new_parameters.append(self._var_key_param)
        # print(new_parameters)
        return self._signature.replace(parameters=new_parameters)

    def merge_arguments(self, new_bound_args):        
        args = []
        kwargs = {}
        all_bound_args = dict(itertools.chain(
            self._bound_args.arguments.items(), new_bound_args.arguments.items()
        ))
        for name, param in self._signature.parameters.items():
            if param.kind != inspect.Parameter.VAR_POSITIONAL \
                    and param.kind != inspect.Parameter.VAR_KEYWORD:
                if param.default == inspect.Parameter.empty:
                    args.append(all_bound_args[name])
                elif name in all_bound_args:
                    kwargs[name] = all_bound_args[name]
                else:
                    kwargs[name] = param.default
        if self._var_pos_param:
            if self._var_pos_param.name in self._bound_args.arguments:
                args.extend(self._bound_args.arguments[self._var_pos_param.name])
            if self._var_pos_param.name in new_bound_args.arguments:
                args.extend(new_bound_args.arguments[self._var_pos_param.name])
        if self._var_key_param:
            if self._var_key_param.name in self._bound_args.arguments:
                kwargs.update(self._bound_args.arguments[self._var_key_param.name])
            if self._var_key_param.name in new_bound_args.arguments:
                kwargs.update(new_bound_args.arguments[self._var_key_param.name])
        return args, kwargs