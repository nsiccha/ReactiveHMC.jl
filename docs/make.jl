using Documenter, DocumenterVitepress, ReactiveHMC

makedocs(
    sitename = "ReactiveHMC.jl",
    modules  = [ReactiveHMC],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveHMC.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"      => "index.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveHMC.jl",
    devbranch = "dev",
    push_preview = true,
)
