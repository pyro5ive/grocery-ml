from abstractions.services.item_id_index_service_base import ItemIndexBuilderServiceBase
from services.item_index_service.item_index_service import ItemIndexBuilderService


class ItemIndexBuilderServiceFactory:
    def __init__(self) -> None:
        pass

    def create(self) -> ItemIndexBuilderServiceBase:
        return ItemIndexBuilderService(existing_mapping=None)