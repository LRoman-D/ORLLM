from __future__ import annotations

from steporlm_stage1.templates.assignment import AssignmentTemplate
from steporlm_stage1.templates.knapsack import KnapsackTemplate
from steporlm_stage1.templates.production_planning import ProductionPlanningTemplate
from steporlm_stage1.templates.resource_allocation import ResourceAllocationTemplate
from steporlm_stage1.templates.tsp import TSPTemplate


TEMPLATE_REGISTRY = {
    "resource_allocation": ResourceAllocationTemplate(),
    "production_planning": ProductionPlanningTemplate(),
    "assignment": AssignmentTemplate(),
    "knapsack": KnapsackTemplate(),
    "tsp": TSPTemplate(),
}
