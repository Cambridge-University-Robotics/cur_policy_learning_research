from dm_control import mjcf

def get_full_identifier(element: mjcf.base.Element):
    if element.parent_model is None:
        return element.name
    return mjcf.get_attachment_frame(element).full_identifier + element.name
