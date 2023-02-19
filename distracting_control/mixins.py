class GetStateMixin:
    def get_distracting_state(self):
        # Get all public attributes in a class except for _env
        state = {key:val for key, val in vars(self).items() if not callable(getattr(self, key)) and not key.startswith('__')}
        state.pop('_env')
        return state
