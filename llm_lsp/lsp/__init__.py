from os import path

from logzero import logger
from lsprotocol.types import *  # noqa: F403
from pygls.lsp.client import BaseLanguageClient


def find_venv(root):
    dir = path.join(root, "venv")
    if path.exists(dir):
        return dir
    # TODO: add more


async def create_lsp_for_language(language: str, directory: str):
    if language == "python":
        return await create_python_lsp(directory)


async def create_python_lsp(directory):
    lsp: BaseLanguageClient = BaseLanguageClient("pylsp", "1.0.0")
    await lsp.start_io("pylsp")
    logger.info("Now initializing")

    @lsp.feature("workspace/configuration")
    def configuration(ls, params):
        logger.debug("It wants configuration!!!")

    @lsp.feature("textDocument/publishDiagnostics")
    def diagnostics(ls, params):
        # pass
        diagnostics = [
            d
            for d in params.diagnostics
            if d.tags and DiagnosticTag.Deprecated in d.tags
        ]
        if len(diagnostics) > 0:
            logger.debug(diagnostics)
        # logger.debug(params)

    _ = await lsp.initialize_async(
        InitializeParams(
            root_path=directory,
            capabilities=ClientCapabilities(
                workspace=WorkspaceClientCapabilities(
                    configuration=True,
                    did_change_configuration=DidChangeConfigurationClientCapabilities(
                        dynamic_registration=True
                    ),
                    workspace_folders=True,
                ),
                text_document=TextDocumentClientCapabilities(
                    completion=CompletionClientCapabilities(
                        completion_item=CompletionClientCapabilitiesCompletionItemType(
                            snippet_support=True,
                            deprecated_support=True,
                            documentation_format=["markdown", "plaintext"],
                            preselect_support=True,
                            label_details_support=True,
                            resolve_support=CompletionClientCapabilitiesCompletionItemTypeResolveSupportType(
                                properties=[
                                    "deprecated",
                                    "documentation",
                                    "detail",
                                    "additionalTextEdits",
                                ]
                            ),
                            tag_support=CompletionClientCapabilitiesCompletionItemTypeTagSupportType(
                                value_set=[CompletionItemTag.Deprecated]
                            ),
                            insert_replace_support=True,
                        )
                    ),
                    publish_diagnostics=PublishDiagnosticsClientCapabilities(
                        tag_support=PublishDiagnosticsClientCapabilitiesTagSupportType(
                            value_set=[DiagnosticTag.Deprecated]
                        )
                    ),
                ),
            ),
        )
    )
    # logger.debug(initialize_result)

    lsp.initialized(InitializedParams())
    logger.info(
        "Using python: "
        + path.abspath(path.join(find_venv(directory), "bin", "python"))
    )
    lsp.workspace_did_change_configuration(
        DidChangeConfigurationParams(
            settings={
                "pylsp.plugins.jedi.environment": path.abspath(
                    path.join(find_venv(directory), "bin", "python")
                ),
                "pylsp.plugins.jedi_completion.include_class_objects": True,
                "pylsp.plugins.jedi_completion.include_function_objects": True,
                "pylsp.plugins.rope_completion.enabled": True,
            }
        )
    )
    return lsp
