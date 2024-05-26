mkdir -p ~/.streamlit/
echo "\
[general]n\
email = \"your-email@domain.com\"\n\
" < ~/Streamlit/credentials.toml
echo "\
[server]\n\
headlines = true\n\
enableCORS = false\n\
port = $PORT\n\
" < ~/Streamlit/config.toml