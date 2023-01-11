# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List
import logging
import traceback
from dataclasses import dataclass
import argparse
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.management import ServiceBusAdministrationClient
from azure.servicebus.exceptions import SessionLockLostError
from azure.core.exceptions import ResourceExistsError
import json
import datetime
import os
from azure.identity import ManagedIdentityCredential


@dataclass
class MultiNodeConfig:
    world_size: int = 1
    world_rank: int = 0
    multinode_available: bool = False
    main_node: bool = True


class ServiceBusMPILikeDriver:
    COMM_TAG_HEAD_ADDRESS = 42
    COMM_TAG_WORKER_ADDRESS = 43

    def __init__(
        self,
        world_size: int,
        world_rank: int,
        topic: str,
        subscription: str,
        allowed_tags=[42, 43],
        sb_host: str = None,
        auth_method: str = "ManagedIdentity",
    ):
        self.logger = logging.getLogger(__name__)

        self.topic = topic

        if subscription is None:
            try:
                from azureml.core import Run

                run = Run.get_context()
                self.subscription = run.parent.id
            except:
                raise Exception(
                    "subscription was left unspecified, and azureml.core.Run cannot get imported."
                )
        else:
            self.subscription = subscription

        # resolve auth method
        self.auth_method = auth_method
        self.sb_host = sb_host
        if self.sb_host is None and self.auth_method == "ManagedIdentity":
            raise Exception(
                "sb_host must be specified when using ManagedIdentity auth."
            )
        if self.auth_method == "ManagedIdentity":
            if "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
                self.logger.info(
                    "Using default identity client id {}".format(
                        os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
                    )
                )
                self.auth_credential = ManagedIdentityCredential(
                    client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
                )
            else:
                self.auth_credential = ManagedIdentityCredential()
        elif self.auth_method == "ConnectionString":
            try:
                from azureml.core import Run, Workspace
                from azureml.core.keyvault import Keyvault

                run = Run.get_context()
                workspace = run.experiment.workspace
                kv = workspace.get_default_keyvault()
                self.connection_str = kv.get_secret("MCDSERVICEBUSCONNSTR")
            except Exception as e:
                self.logger.warning("Exception during kv access: {}".format(e))
                self.connection_str = None

            if self.connection_str is None:
                self.logger.warning(
                    "secret MCDSERVICEBUSCONNSTR not found in the workspace default keyvault, using env var instead which is NOT SECURE."
                )
                if "MCDSERVICEBUSCONNSTR" not in os.environ:
                    raise Exception("MCDSERVICEBUSCONNSTR not found in env var either.")
                self.connection_str = os.environ["MCDSERVICEBUSCONNSTR"]
        else:
            raise Exception("Unknown auth_method {}".format(self.auth_method))

        self.mgmt_client = None
        self.clients = {}
        self.senders = {}
        self.receivers = {}
        self.allowed_tags = allowed_tags

        # we can't auto-detect
        self.multinode_config = MultiNodeConfig(
            world_size=world_size,
            world_rank=world_rank,
            multinode_available=(world_size > 1),
            main_node=(world_rank == 0),
        )

    def get_subscription(self):
        self.logger.info("Locating subscription {}".format(self.subscription))

        for subscription_properties in self.mgmt_client.list_subscriptions(self.topic):
            if subscription_properties.name == self.subscription:
                self.logger.info("Found subscription {}".format(self.subscription))
                return subscription_properties
        else:
            return None

    def create_subscription(self):
        self.logger.info("Creating subscription {}".format(self.subscription))
        try:
            self.mgmt_client.create_subscription(
                self.topic,
                self.subscription,
                requires_session=True,
                default_message_time_to_live=datetime.timedelta(minutes=10),
                max_delivery_count=2000,
                auto_delete_on_idle=datetime.timedelta(minutes=60),
            )
        except ResourceExistsError:
            # to avoid concurrent creation
            self.logger.info("Subscription already exists")

    # session id management

    def get_session_key(self, source: int, target: int, tag=None):
        return "{}=>{}:{}".format(source, target, tag or "*")

    def _initialize_session(self, source: int, target: int, tag: str = None):
        if source == target:
            raise Exception("source and target must be different")
        _session_key = self.get_session_key(source, target, tag)
        self.logger.info("Initializing session {}".format(_session_key))

        if source == self.multinode_config.world_rank:
            self.senders[_session_key] = self.clients[_session_key].get_topic_sender(
                topic_name=self.topic, session_id=_session_key
            )
            self.logger.info("Opening sender {}".format(_session_key))
            self.senders[_session_key].__enter__()
        if target == self.multinode_config.world_rank:
            self.receivers[_session_key] = self.clients[
                _session_key
            ].get_subscription_receiver(
                topic_name=self.topic,
                subscription_name=self.subscription,
                max_wait_time=5,
                session_id=_session_key,
            )
            self.logger.info("Opening receiver {}".format(_session_key))
            self.receivers[_session_key].__enter__()

    def _reinitialize_session(self, source: int, target: int, tag=None):
        # for when SessionLockLostError happens
        _session_key = self.get_session_key(source, target, tag)
        self.logger.info("Re-initializing session {}".format(_session_key))
        self.clients[_session_key].__exit__()
        self._initialize_client(source, target, tag=tag)

        if _session_key in self.senders:
            self.senders[_session_key].__exit__()
            del self.senders[_session_key]
        if _session_key in self.receivers:
            self.receivers[_session_key].__exit__()
            del self.receivers[_session_key]
        self._initialize_session(source, target, tag=tag)

    def _initialize_client(self, source: int, target: int, tag=None):
        _session_key = self.get_session_key(source, target, tag)
        self.logger.debug("Creating client {}".format(_session_key))
        if self.auth_method == "ManagedIdentity":
            self.clients[_session_key] = ServiceBusClient(
                fully_qualified_namespace=self.sb_host,
                credential=self.auth_credential,
                session_id=_session_key,
            )
        elif self.auth_method == "ConnectionString":
            self.clients[_session_key] = ServiceBusClient.from_connection_string(
                conn_str=self.connection_str,
                logging_enable=True,
                session_id=_session_key,
            )
        self.logger.debug("Opening client {}".format(_session_key))
        self.clients[_session_key].__enter__()

    def initialize(self):
        """Initialize the driver"""
        self.logger.info(f"Call to {self.__class__.__name__}.initialize()")

        if self.auth_method == "ManagedIdentity":
            self.mgmt_client = ServiceBusAdministrationClient(
                fully_qualified_namespace=self.sb_host, credential=self.auth_credential
            )
        elif self.auth_method == "ConnectionString":
            self.mgmt_client = ServiceBusAdministrationClient.from_connection_string(
                conn_str=self.connection_str
            )

        if self.get_subscription() is None:
            # first node to execute this will create the subscription for others to join
            self.create_subscription()
            # alternatively:
            # if self.multinode_config.main_node:
            #     self.create_subscription()
            # else:
            #     raise Exception("Subscription {} does not exist".format(self.subscription))

        for source in range(self.multinode_config.world_size):
            for target in range(self.multinode_config.world_size):
                for tag in self.allowed_tags:
                    if source == target:
                        continue
                    self._initialize_client(source, target, tag=tag)

    def finalize(self):
        """Finalize/close resources used by the driver"""
        self.logger.info(f"Call to {self.__class__.__name__}.finalize()")
        for key in self.senders:
            self.logger.info("Closing sender {}".format(key))
            self.senders[key].__exit__()
        for key in self.receivers:
            self.logger.info("Closing receiver {}".format(key))
            self.receivers[key].__exit__()
        for key in self.clients:
            self.logger.info("Closing client {}".format(key))
            self.clients[key].__exit__()

    def get_multinode_config(self) -> MultiNodeConfig:
        """Get internal multinode config"""
        if self.multinode_config:
            return self.multinode_config
        else:
            raise Exception("Multinode config is None, use initialize() first.")

    def get_comm(self):
        """Returns the communicator"""
        return self

    # fake comm methods
    def recv(self, source: int, tag: str = None, blocking=True):
        self.logger.info("Listening to {}".format(source))
        _session_key = self.get_session_key(
            source, self.multinode_config.world_rank, tag=tag
        )
        if _session_key not in self.receivers:
            self._initialize_session(source, self.multinode_config.world_rank, tag=tag)

        # safe receive
        received_message = False
        retries = 0
        while received_message is False and retries < 10:
            try:
                received_msgs = self.receivers[_session_key].receive_messages(
                    max_message_count=10, max_wait_time=5
                )
                for msg in received_msgs:
                    self.logger.info("Received message from {}".format(_session_key))
                    received_message = True
                    self.receivers[_session_key].complete_message(msg)
                    return json.loads(str(msg))
                if blocking:
                    self.logger.info(
                        "No message from {}, waiting...".format(_session_key)
                    )
                else:
                    return None
                retries = 0  # reset retries, because no issue happened here.
            except SessionLockLostError:
                self.logger.warning(
                    "SessionLockLostError: The lock on the session has expired. Callers should request the session again."
                )
                self._reinitialize_session(
                    source, self.multinode_config.world_rank, tag=tag
                )
                retries += 1
            except BaseException as e:
                self.logger.info(
                    "receive_messages() on receiver {} excepted: {}".format(
                        _session_key, traceback.format_exc()
                    )
                )
                retries += 1

        raise Exception(
            "Exhausted retries during receive_messages() on receiver {}".format(
                _session_key
            )
        )

    def flush_recv(self):
        for key in self.receivers:
            self.logger.info("Flushing receiver {}".format(key))
            try:
                for msg in self.receivers[key]:
                    self.logger.info("Flushing message from {}".format(key))
                    self.receivers[key].complete_message(msg)
            except BaseException as e:
                self.logger.info(
                    "Flushing receiver {} excepted: {}".format(
                        key, traceback.format_exc()
                    )
                )

    def send(self, message, target: int, tag: str = None):
        self.logger.info("Sending message to {}".format(target))
        _session_key = self.get_session_key(
            self.multinode_config.world_rank, target, tag=tag
        )
        if _session_key not in self.senders:
            self._initialize_session(self.multinode_config.world_rank, target, tag=tag)

        sb_message = ServiceBusMessage(json.dumps(message), session_id=_session_key)
        self.logger.info("Sending message via {}".format(_session_key))
        self.senders[_session_key].send_messages(sb_message)

    # we can't have a non-blocking here, but recv timeout is 5s.
    def iprobe(self, source: int, tag: str = None):
        return self.recv(source, tag=tag, blocking=False)
