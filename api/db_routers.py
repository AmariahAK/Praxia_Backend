class ShardingRouter:
    """
    A router to control database sharding based on user ID
    """
    
    def db_for_read(self, model, **hints):
        """
        Reads go to the appropriate shard based on user ID
        """
        if hasattr(hints.get('instance', None), 'user_id'):
            user_id = hints['instance'].user_id
            return self._get_shard_for_user(user_id)
        return 'default'
    
    def db_for_write(self, model, **hints):
        """
        Writes go to the appropriate shard based on user ID
        """
        if hasattr(hints.get('instance', None), 'user_id'):
            user_id = hints['instance'].user_id
            return self._get_shard_for_user(user_id)
        return 'default'
    
    def allow_relation(self, obj1, obj2, **hints):
        """
        Relations are allowed if both objects are in the same shard
        """
        if hasattr(obj1, 'user_id') and hasattr(obj2, 'user_id'):
            shard1 = self._get_shard_for_user(obj1.user_id)
            shard2 = self._get_shard_for_user(obj2.user_id)
            return shard1 == shard2
        return True
    
    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        All models can be migrated to all databases
        """
        return True
    
    def _get_shard_for_user(self, user_id):
        """
        Determine which shard to use based on user ID
        """
        # Simple modulo-based sharding
        shard_count = 3  # Number of shards
        shard_index = user_id % shard_count
        
        if shard_index == 0:
            return 'default'
        else:
            return f'shard_{shard_index}'
