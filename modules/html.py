from html import escape

from modules.localization import translate

progress_html = '''
<div class="loader-container">
  <div class="loader"></div>
  <div class="progress-container">
    <progress value="*number*" max="100"></progress>
  </div>
  <span>*text*</span>
</div>
'''


def make_progress_html(number, text, params=None):
    return progress_html.replace('*number*', str(number)).replace('*text*', escape(translate(text, params)))
