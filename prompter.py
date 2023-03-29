import os
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, meta


class Prompter:
    def __init__(
        self,
        model,
        templates_path: Optional[str] = None,
        allowed_missing_variables: Optional[List[str]] = None,
        default_variable_values: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.templates_path = self.get_templates_path(templates_path)
        self.environment = Environment(loader=FileSystemLoader(self.templates_path))
        self.model = model

        self.allowed_missing_variables = allowed_missing_variables or [
            "examples",
            "description",
            "output_format",
        ]
        self.default_variable_values = default_variable_values or {}
        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[1 : self.model_args_count]
        self.prompt_variables_map = {}

    def get_templates_path(self, templates_path: Optional[str]) -> str:
        if templates_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            templates_dir = os.path.join(dir_path, "templates")
        else:
            self.verify_template_path(templates_path)
            templates_dir = templates_path
        return templates_dir

    def verify_template_path(self, templates_path: str):
        if not os.path.exists(templates_path):
            raise ValueError(f"Templates path {templates_path} does not exist")

    def list_templates(self) -> List[str]:
        """Returns a list of available templates."""
        return self.environment.list_templates()

    def get_template_variables(self, template_name: str) -> List[str]:
        """Returns a list of undeclared variables in the template."""

        if template_name in self.prompt_variables_map:
            return self.prompt_variables_map[template_name]
        template_source = self.environment.loader.get_source(self.environment, template_name)
        parsed_content = self.environment.parse(template_source)
        undeclared_variables = meta.find_undeclared_variables(parsed_content)
        self.prompt_variables_map[template_name] = undeclared_variables
        return undeclared_variables

    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """Generates a prompt using the given template and keyword arguments."""

        variables = self.get_template_variables(template_name)
        variables_missing = [
            variable
            for variable in variables
            if variable not in kwargs
            and variable not in self.allowed_missing_variables
            and variable not in self.default_variable_values
        ]

        if variables_missing:
            raise ValueError(
                f"Missing required variables in template {', '.join(variables_missing)}"
            )

        kwargs.update(self.default_variable_values)
        template = self.environment.get_template(template_name)
        prompt = template.render(**kwargs).strip()
        return prompt

    def fit(self, template_name: str, **kwargs):
        """Runs the model with the prompt generated from the given template and keyword arguments."""

        prompt_variables = self.get_template_variables(template_name)
        prompt_kwargs = {
            variable: value
            for variable, value in kwargs.items()
            if variable in prompt_variables
        }

        prompt = self.generate_prompt(template_name, **prompt_kwargs)
        output = self.model.execute_with_retry(prompts=[prompt])
        output = self.model_output(output)
        return output

    def model_output(self, raw_output: Dict[str, Any]) -> Any:
        """
        Process the raw output from the model and return the result in the desired format.
        Override this method to implement custom output processing.
        """
        return raw_output.get("choices", [])[0].get("text", "").strip()
