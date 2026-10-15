from __future__ import annotations

from fastapi import Request

from starling.config import Settings
from starling.gpu.placer import GpuPlacer
from starling.lifecycle.manager import ModelManager
from starling.registry.catalog import ModelRegistry


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_placer(request: Request) -> GpuPlacer:
    return request.app.state.placer
