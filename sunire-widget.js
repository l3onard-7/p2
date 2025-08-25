(function() {
    // Prevent multiple widget loads
    if (window.SunireWidget) return;
    
    // Get script tag and extract configuration
    const scriptTag = document.currentScript;
    const widgetId = scriptTag.getAttribute('data-widget-id') || 'default';
    const backendUrl = scriptTag.getAttribute('data-backend-url') || 'https://p2-sand-phi.vercel.app';
    const theme = scriptTag.getAttribute('data-theme') || 'default';
    const position = scriptTag.getAttribute('data-position') || 'bottom-right';

    // Widget configuration
    const config = {
        widgetId: widgetId,
        backendUrl: backendUrl,
        theme: theme,
        position: position
    };

    // Create widget container and iframe
    function createWidget() {
        const widgetContainer = document.createElement('div');
        widgetContainer.id = 'sunire-chatbot-widget';
        
        // Position styles based on config
        const positionStyles = {
            'bottom-right': 'bottom: 20px; right: 20px;',
            'bottom-left': 'bottom: 20px; left: 20px;',
            'top-right': 'top: 20px; right: 20px;',
            'top-left': 'top: 20px; left: 20px;'
        };
        
        widgetContainer.style.cssText = `
            position: fixed;
            ${positionStyles[position] || positionStyles['bottom-right']}
            z-index: 999999;
            width: 400px;
            height: 600px;
            max-width: 90vw;
            max-height: 90vh;
            border: none;
            background: transparent;
            pointer-events: none;
        `;

        // Create iframe to load your React app
        const iframe = document.createElement('iframe');
        iframe.id = 'sunire-chat-iframe';
        iframe.src = `https://p2-frontend.vercel.app?widget=true&backend=${encodeURIComponent(backendUrl)}&theme=${theme}&widgetId=${widgetId}`;
        iframe.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            background: transparent;
            pointer-events: auto;
        `;
        iframe.allow = "clipboard-write";

        widgetContainer.appendChild(iframe);
        document.body.appendChild(widgetContainer);

        // Handle iframe messages for resizing
        window.addEventListener('message', function(event) {
            if (event.origin !== 'https://p2-frontend.vercel.app') return;
            
            if (event.data.type === 'SUNIRE_WIDGET_RESIZE') {
                widgetContainer.style.width = event.data.width + 'px';
                widgetContainer.style.height = event.data.height + 'px';
            }
        });

        return widgetContainer;
    }

    // Initialize widget
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', createWidget);
        } else {
            createWidget();
        }
    }

    // Expose widget API
    window.SunireWidget = {
        init: init,
        config: config,
        destroy: function() {
            const widget = document.getElementById('sunire-chatbot-widget');
            if (widget) {
                widget.remove();
            }
            delete window.SunireWidget;
        }
    };

    // Auto-initialize
    init();
})();