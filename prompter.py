import os
import glob
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, meta


class Prompter:
    def __init__(
        self,
        model,
        template: Optional[str] = None,
        allowed_missing_variables: Optional[List[str]] = None,
        default_variable_values: Optional[Dict[str, Any]] = None,
        max_completion_length: int = 20,
    ) -> None:
        
        
        assert template!=None,"ReferenceError: template is not defined"
        
        self.load_template(template)
        self.model = model
        self.max_completion_length = max_completion_length

        self.allowed_missing_variables = allowed_missing_variables or [
            "examples",
            "description",
            "output_format",
        ]
        self.default_variable_values = default_variable_values or {}
        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[1 : self.model_args_count]
        self.prompt_variables_map = {}


    def availabel_templates(self, template_path: str) -> Dict[str, str]:

        all_templates = glob.glob(f'{template_path}/*jinja')
        template_names= [template.split('/')[-1] for template in all_templates]
        template_dict = dict(zip(template_names, all_templates))
        return template_dict
    

    def load_template(self, template: str):
        
        dir_path          = os.path.dirname(os.path.realpath('./codes/'))
        
        
        templates_dir     = os.path.join(dir_path, "templates")
        print(templates_dir)
        
        default_templates = self.availabel_templates(templates_dir)
        

        if template in default_templates:

            self.template_name = template
            self.template_dir  = templates_dir
            self.environment   = Environment(loader=FileSystemLoader(templates_dir))
            self.template      = self.environment.get_template(template)

        else:

            self.verify_template_path(template)
            custom_template_dir, custom_template_name = os.path.split(template)

            self.template_name = custom_template_name
            self.template_dir  = custom_template_dir
            self.environment   = Environment(loader=FileSystemLoader(custom_template_dir))
            self.template      = self.environment.get_template(custom_template_name)

        return self.template
    

    def verify_template_path(self, templates_path: str):
        if not os.path.exists(templates_path):
            raise ValueError(f"Templates path {templates_path} does not exist")

    def list_templates(self) -> List[str]:
        """Returns a list of available templates."""
        return self.environment.list_templates()

    def get_template_variables(self) -> List[str]:
        """Returns a list of undeclared variables in the template."""

        if self.template_name in self.prompt_variables_map:
            return self.prompt_variables_map[self.template_name]
        
        template_source = self.environment.loader.get_source(self.environment, self.template_name)
        parsed_content = self.environment.parse(template_source)
        undeclared_variables = meta.find_undeclared_variables(parsed_content)
        self.prompt_variables_map[self.template_name] = undeclared_variables
        return undeclared_variables

    def generate_prompt(self, **kwargs) -> str:
        """Generates a prompt using the given template and keyword arguments."""

        variables = self.get_template_variables()
        
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
        prompt = self.template.render(**kwargs).strip()
        return prompt
    

    def raw_fit(self, prompt: str):
        """Runs the model with the given prompt."""
        
        outputs = [self.model.model_output_raw(k) for output in self.model.run(prompts=[prompt])]
        return outputs


    def fit(self, **kwargs):
        
        """Runs the model with the prompt generated from the given template and keyword arguments."""
        
        
        prompt_variables = self.get_template_variables()
        prompt_kwargs = {
            variable: value
            for variable, value in kwargs.items()
            if variable in prompt_variables
        }
        prompt = self.generate_prompt(**prompt_kwargs)
        response = self.model.execute_with_retry(prompts=[prompt])
        outputs  = [self.model.model_output(output, max_completion_length = self.max_completion_length) for output in response]
        return outputs
