cpu_memory_html = """
    <html>
        <head>
            <meta charset="utf-8">
            <title></title>
            <style type="text/css">
                #progress{{
                    height: 30px;
                    border: 1px solid #ccc;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 0 5px 0px #ddd inset;
                }}
                #progress span{{
                    display: inline-block;
                    height: 30px;
                    line-height: 30px;
                    background: #2989d8;
                    text-align: right;
                    box-sizing: border-box;
                    padding-right: 5px;
                    color: #000000;
                }}
            </style>
        </head>
        <body>
            <div>
                <div id="progress">
                    <span style="width: {0}%;">{0}%</span>
                </div>
                <div style='height:25px; width:100%';></div>
                <div id="progress">
                    <span style="width: {1}%;">{1}%</span>
                </div>
            </div>
        </body>
    </html>
"""


ray_status_html = """
    <html>
        <head>
            <meta charset="utf-8">
            <title></title>
            <style type="text/css">
                #progress{{
                    height: 30px;
                    width: 100%;
                    border: 1px solid #ccc;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 0 5px 0px #ddd inset;
                }}
                #progress span{{
                    display: inline-block;
                    height: 30px;
                    line-height: 30px;
                    background: #2989d8;
                    text-align: right;
                    box-sizing: border-box;
                    padding-right: 5px;
                    color: #fff;
                }}
            </style>
        </head>
        <body>
            <div>
                <div id="progress">
                    <span style="width: {0}%;">{1}/{2}</span>
                </div>
            </div>
        </body>
    </html>
"""


custom_css = """
OpenChatKit .overflow-y-auto{height:500px}

#notshowing {align-content: left;}
#notshowimg div.icon-buttons {display: none !important}
#notice_markdown th {
    display: none;
}

.notice_markdown {
    text-align: center;
    background: #2e78c4;
    padding: 1%;
    height: 5rem;
    color: #fff !important;
    margin-top: 0;
    border-radius: 10px;
}

.disablegenerating div:first-child {
    border: none !important
}

gradio-app {
    background: linear-gradient(to bottom, #86ccf5, #3273bf) !important;
    padding: 3%;
}

.gradio-container {
    margin: 0 auto !important;
    padding: 2% !important;
    background: #fff !important;
    border-radius: 10px !important;
}

footer {
    display: none !important;
}

.footer {
    margin-top: 2rem !important;
    text-align: center;
    border-bottom: 1px solid #e5e5e5;
}

.footer>p {
    font-size: .8rem;
    display: inline-block;
    padding: 0 10px;
    transform: translateY(10px);
    background: white;
}

#tabcontent div.gap div:first-child{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: nowrap;
}

#tabcontent div.gap div:first-child div{
    flex: 1;
}

#botinline{
    display:inline !important
}


.statusstyle {
    min-height: 0px !important;
    background: white;
    position: absolute;
    z-index: 99999;
    margin-top: 6px;
    width: 75px;
    height: 30px;
    left: 50%;
    top: 50%;
    transform: translate(-50%,-50%);
}

.statusstyle>p {
    text-align: center;
    border: solid 1px;
    border-radius: 6px;
    padding: 3px 8px;
}

.statusstyle div:first-child{
    border: none !important
}

.btn-style {
    background: #3c94dc !important;
    color: #ffffff !important;
    position: absolute;
    height: 30px;
    align-content: center;
    min-width: 0px !important;
    width: 75px;
    border-radius: 6px !important;
    left: 50%;
    top: 50%;
    transform: translate(-50%,-20%);
}

.btn-style:hover{
    box-shadow: #52a0e0 !important;
    background: #52a0e0 !important;


}

.btn-style:active {
    background: #3c94dc !important;
}
"""
